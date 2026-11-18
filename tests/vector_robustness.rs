//! Robustness tests for vector tessellation: sharp joins, degenerate
//! segments, empty clips, disjoint subpaths and huge coordinates.
//!
//! Scenes are fixed SVG path strings. Rendering happens on the
//! software backend (llvmpipe) and is compared against checked-in
//! golden images in `tests/golden/`. Regenerate goldens with:
//!
//!     VGER_UPDATE_GOLDEN=1 cargo test --test vector_robustness

use futures::executor::block_on;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vger::color::Color;
use vger::defs::*;
use vger::*;

mod common;
use common::*;

fn setup() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let (device, queue) = block_on(common::setup());
    (Arc::new(device), Arc::new(queue))
}

fn make_vger() -> (Vger, Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let (device, queue) = setup();
    let vger = Vger::new(
        device.clone(),
        queue.clone(),
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );
    (vger, device, queue)
}

// ---------------------------------------------------------------------------
// Minimal SVG path data parser (M L H V Q Z, absolute and relative).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum PathCmd {
    Move(LocalPoint),
    Line(LocalPoint),
    Quad(LocalPoint, LocalPoint),
    Close,
}

fn tokenize_path(d: &str) -> Vec<String> {
    let mut tokens = vec![];
    let mut cur = String::new();
    for ch in d.chars() {
        if ch.is_ascii_alphabetic() {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            tokens.push(ch.to_string());
        } else if ch.is_whitespace() || ch == ',' {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        } else if ch == '-' {
            // A minus sign always starts a new number (we don't use
            // exponent notation in test data).
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            cur.push(ch);
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

fn parse_svg_path(d: &str) -> Vec<PathCmd> {
    let tokens = tokenize_path(d);
    let mut cmds = vec![];
    let mut i = 0;
    let mut cur = (0.0f32, 0.0f32);
    let mut cmd = ' ';

    let num = |tok: &String| -> f32 {
        tok.parse::<f32>()
            .unwrap_or_else(|_| panic!("bad number in SVG path: {:?}", tok))
    };

    while i < tokens.len() {
        if tokens[i].len() == 1 && tokens[i].chars().next().unwrap().is_ascii_alphabetic() {
            cmd = tokens[i].chars().next().unwrap();
            i += 1;
            if cmd == 'Z' || cmd == 'z' {
                cmds.push(PathCmd::Close);
                continue;
            }
        }
        let rel = cmd.is_ascii_lowercase();
        let mut pt = |x: f32, y: f32| -> (f32, f32) {
            if rel {
                (cur.0 + x, cur.1 + y)
            } else {
                (x, y)
            }
        };
        match cmd.to_ascii_uppercase() {
            'M' => {
                let p = pt(num(&tokens[i]), num(&tokens[i + 1]));
                cmds.push(PathCmd::Move(p.into()));
                cur = p;
                i += 2;
            }
            'L' => {
                let p = pt(num(&tokens[i]), num(&tokens[i + 1]));
                cmds.push(PathCmd::Line(p.into()));
                cur = p;
                i += 2;
            }
            'H' => {
                let x = num(&tokens[i]);
                let p = if rel { (cur.0 + x, cur.1) } else { (x, cur.1) };
                cmds.push(PathCmd::Line(p.into()));
                cur = p;
                i += 1;
            }
            'V' => {
                let y = num(&tokens[i]);
                let p = if rel { (cur.0, cur.1 + y) } else { (cur.0, y) };
                cmds.push(PathCmd::Line(p.into()));
                cur = p;
                i += 1;
            }
            'Q' => {
                let b = pt(num(&tokens[i]), num(&tokens[i + 1]));
                let c = pt(num(&tokens[i + 2]), num(&tokens[i + 3]));
                cmds.push(PathCmd::Quad(b.into(), c.into()));
                cur = c;
                i += 4;
            }
            other => panic!("unsupported SVG path command: {}", other),
        }
    }
    cmds
}

/// Feeds SVG path data into vger. Lines become degenerate quadratic
/// segments (control point on the chord), exactly the join geometry
/// that used to produce NaN coverage.
fn draw_svg_path(vger: &mut Vger, d: &str) {
    let mut pen = LocalPoint::zero();
    let mut subpath_start = LocalPoint::zero();
    for cmd in parse_svg_path(d) {
        match cmd {
            PathCmd::Move(p) => {
                vger.move_to(p);
                pen = p;
                subpath_start = p;
            }
            PathCmd::Line(p) => {
                let mid = pen.lerp(p, 0.5);
                vger.quad_to(mid, p);
                pen = p;
            }
            PathCmd::Quad(b, c) => {
                vger.quad_to(b, c);
                pen = c;
            }
            PathCmd::Close => {
                if pen != subpath_start {
                    let mid = pen.lerp(subpath_start, 0.5);
                    vger.quad_to(mid, subpath_start);
                    pen = subpath_start;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Golden image comparison.
// ---------------------------------------------------------------------------

struct Image {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Image {
    /// Pixel at local (vger) coordinates; vger's y axis is flipped
    /// relative to image rows.
    fn pixel_local(&self, x: usize, y: usize) -> [u8; 4] {
        let row = self.height - 1 - y;
        let o = (row * self.width + x) * 4;
        [self.data[o], self.data[o + 1], self.data[o + 2], self.data[o + 3]]
    }

    fn is_black(&self) -> bool {
        // Ignore alpha: the render target is cleared to opaque black.
        self.data.chunks(4).all(|px| px[..3] == [0, 0, 0])
    }
}

fn load_png(path: &str) -> Image {
    let decoder = png::Decoder::new(File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgba);
    Image {
        width: info.width as usize,
        height: info.height as usize,
        data: buf[..info.buffer_size()].to_vec(),
    }
}

const GOLDEN_DIR: &str = "tests/golden";
const OUT_DIR: &str = "target/vector_robustness";

/// Renders the scene and compares it against the checked-in golden.
/// Allows tiny per-pixel differences (<= 2/255 on a small fraction of
/// pixels) to absorb minor software-rasterizer version differences.
fn assert_golden(vger: &mut Vger, device: &wgpu::Device, queue: &wgpu::Queue, name: &str) -> Image {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    std::fs::create_dir_all(GOLDEN_DIR).unwrap();

    let out_path = format!("{}/{}", OUT_DIR, name);
    render_test(vger, device, queue, &out_path, false);

    let golden_path = format!("{}/{}", GOLDEN_DIR, name);

    if std::env::var("VGER_UPDATE_GOLDEN").is_ok() {
        std::fs::copy(&out_path, &golden_path).unwrap();
        println!("updated golden {}", golden_path);
    }

    assert!(
        Path::new(&golden_path).exists(),
        "missing golden {} — run with VGER_UPDATE_GOLDEN=1 to create it",
        golden_path
    );

    let actual = load_png(&out_path);
    let expected = load_png(&golden_path);
    assert_eq!(actual.width, expected.width);
    assert_eq!(actual.height, expected.height);
    assert_eq!(actual.data.len(), expected.data.len());

    let mut bad_pixels = 0usize;
    let mut max_diff = 0u8;
    for (a, b) in actual.data.chunks(4).zip(expected.data.chunks(4)) {
        let diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u8)
            .max()
            .unwrap();
        max_diff = max_diff.max(diff);
        if diff > 2 {
            bad_pixels += 1;
        }
    }
    let total = actual.data.len() / 4;
    assert!(
        bad_pixels * 1000 <= total, // <= 0.1% of pixels
        "{}: {} of {} pixels differ from golden (max channel diff {})",
        name,
        bad_pixels,
        total,
        max_diff
    );

    actual
}

fn cyanish(px: [u8; 4]) -> bool {
    px[1] > 200 && px[2] > 200 && px[0] < 100
}

fn is_black(px: [u8; 4]) -> bool {
    // Ignore alpha: the render target is cleared to opaque black.
    px[..3] == [0, 0, 0]
}

/// Asserts every prim bbox is finite and properly ordered, and returns
/// the overall bounds.
fn assert_bboxes_sane(vger: &Vger, limit: f32) {
    let n = vger.debug_prim_count();
    assert!(n > 0, "expected at least one prim");
    for i in 0..n {
        let b = vger.debug_prim_quad_bounds(i);
        for (j, v) in b.iter().enumerate() {
            assert!(v.is_finite(), "prim {} bound {} is not finite: {}", i, j, v);
            assert!(
                v.abs() <= limit,
                "prim {} bound {} out of range: {}",
                i,
                j,
                v
            );
        }
        assert!(b[0] <= b[2], "prim {} has inverted x bounds: {:?}", i, b);
        assert!(b[1] <= b[3], "prim {} has inverted y bounds: {:?}", i, b);
    }
}

// ---------------------------------------------------------------------------
// Fixed SVG scenes.
// ---------------------------------------------------------------------------

/// Five-pointed star: ten sharp joins (5 outer tips, 5 inner valleys).
const STAR_SVG: &str = "M 256 40 L 310 200 L 480 200 L 340 300 L 390 460 \
                        L 256 360 L 122 460 L 172 300 L 32 200 L 202 200 Z";

/// Two humps meeting in a cusp join at (256, 400).
const CUSP_SVG: &str = "M 60 400 Q 128 80 256 400 Q 384 80 452 400 Z";

/// Fully collinear quadratic chain (degenerate hulls, zero-area fill).
const COLLINEAR_SVG: &str = "M 60 256 Q 133 256 206 256 Q 309 256 412 256 L 452 256 Z";

/// Two disjoint closed squares in one fill command.
const DISJOINT_SVG: &str =
    "M 50 50 L 150 50 L 150 150 L 50 150 Z M 300 300 L 400 300 L 400 400 L 300 400 Z";

/// A small on-screen triangle plus a triangle a million units away.
const HUGE_SVG: &str = "M 100 400 L 200 400 L 150 300 Z \
                        M 1000000 1000000 L 1000100 1000000 L 1000050 1000100 Z";

#[test]
fn svg_star_sharp_joins() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    draw_svg_path(&mut vger, STAR_SVG);
    vger.fill(cyan);

    // Draw command count: 10 segments -> 6 y-bands.
    assert_eq!(vger.debug_prim_count(), 6);
    assert_bboxes_sane(&vger, 512.0);

    let img = assert_golden(&mut vger, &device, &queue, "star.png");

    // Join regions: just inside the top tip, just outside it, and
    // either side of the inner valley at (310, 200).
    assert!(cyanish(img.pixel_local(256, 50)), "inside top tip");
    assert!(is_black(img.pixel_local(256, 30)), "outside top tip");
    assert!(is_black(img.pixel_local(350, 190)), "outside valley notch");
    assert!(cyanish(img.pixel_local(350, 210)), "inside below valley");
    assert!(cyanish(img.pixel_local(256, 280)), "star body");
}

#[test]
fn svg_cusp_joins() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    draw_svg_path(&mut vger, CUSP_SVG);
    vger.fill(cyan);

    // Two quads spanning y in [79, 401] plus the closing chord at
    // y = 400: bands [79, 399] and [399, 401].
    assert_eq!(vger.debug_prim_count(), 2);
    assert_bboxes_sane(&vger, 512.0);

    let img = assert_golden(&mut vger, &device, &queue, "cusp.png");

    // Inside the first hump, above it, and around the cusp join at
    // (256, 400): the humps only touch at the baseline, so the gap
    // directly above the cusp stays empty while both flanks fill.
    assert!(cyanish(img.pixel_local(143, 300)), "inside hump");
    assert!(is_black(img.pixel_local(143, 200)), "above hump");
    assert!(cyanish(img.pixel_local(250, 395)), "left flank of cusp");
    assert!(cyanish(img.pixel_local(262, 395)), "right flank of cusp");
    assert!(is_black(img.pixel_local(256, 395)), "cusp gap stays empty");
    assert!(is_black(img.pixel_local(256, 390)), "above cusp empty");
}

#[test]
fn svg_collinear_degenerate_fill() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    draw_svg_path(&mut vger, COLLINEAR_SVG);
    vger.fill(cyan);

    // All segments share one y band.
    assert_eq!(vger.debug_prim_count(), 1);
    assert_bboxes_sane(&vger, 512.0);

    // Must render without NaN artifacts (golden captures exact output).
    assert_golden(&mut vger, &device, &queue, "collinear.png");
}

#[test]
fn svg_disjoint_subpaths() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    draw_svg_path(&mut vger, DISJOINT_SVG);
    vger.fill(cyan);

    // Three y-bands per square; the empty band between the subpaths
    // must not produce a prim.
    assert_eq!(vger.debug_prim_count(), 6);
    assert_bboxes_sane(&vger, 512.0);

    let img = assert_golden(&mut vger, &device, &queue, "disjoint.png");

    assert!(cyanish(img.pixel_local(100, 100)), "first square filled");
    assert!(cyanish(img.pixel_local(350, 350)), "second square filled");
    // No synthetic connector may join the subpaths: the diagonal
    // between them stays empty.
    assert!(is_black(img.pixel_local(100, 200)), "no connector artifact");
    assert!(is_black(img.pixel_local(200, 200)), "between squares empty");
}

#[test]
fn svg_huge_coordinates() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    draw_svg_path(&mut vger, HUGE_SVG);
    vger.fill(cyan);

    // Two bands per triangle; the ~1e6-unit gap between the subpaths
    // must not produce a prim with a sentinel bounding box.
    assert_eq!(vger.debug_prim_count(), 4);
    assert_bboxes_sane(&vger, 1_100_000.0);

    let img = assert_golden(&mut vger, &device, &queue, "huge.png");

    assert!(cyanish(img.pixel_local(150, 380)), "visible triangle filled");
    assert!(is_black(img.pixel_local(150, 250)), "above triangle empty");
    assert!(is_black(img.pixel_local(400, 100)), "far triangle off-screen");
}

#[test]
fn degenerate_strokes() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);

    // Zero-length segment: round dot, not NaN.
    vger.stroke_segment([100.0, 100.0], [100.0, 100.0], 8.0, cyan);
    // Repeated control point.
    vger.stroke_bezier([300.0, 100.0], [300.0, 100.0], [400.0, 100.0], 8.0, cyan);
    // Fully collinear quadratic.
    vger.stroke_bezier([100.0, 300.0], [200.0, 300.0], [300.0, 300.0], 8.0, cyan);

    assert_eq!(vger.debug_prim_count(), 3);
    assert_bboxes_sane(&vger, 512.0);

    let img = assert_golden(&mut vger, &device, &queue, "degenerate.png");

    assert!(cyanish(img.pixel_local(100, 100)), "zero-length segment dot");
    assert!(is_black(img.pixel_local(100, 110)), "dot falloff");
    assert!(cyanish(img.pixel_local(350, 100)), "repeated-control bezier");
    assert!(cyanish(img.pixel_local(200, 300)), "collinear bezier");
}

#[test]
fn empty_path_fill() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);

    // fill() with no segments: no panic, no prims, no output.
    vger.fill(cyan);
    assert_eq!(vger.debug_prim_count(), 0);

    // move_to without any curve: still nothing to fill.
    vger.move_to([100.0, 100.0]);
    vger.fill(cyan);
    assert_eq!(vger.debug_prim_count(), 0);

    let img = assert_golden(&mut vger, &device, &queue, "empty.png");
    assert!(img.is_black());
}

#[test]
fn empty_scissor() {
    let (mut vger, device, queue) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);

    // A zero-size scissor rect clips everything.
    vger.scissor(euclid::rect(100.0, 100.0, 0.0, 0.0));
    vger.fill_circle([256.0, 256.0], 50.0, cyan);

    let img = assert_golden(&mut vger, &device, &queue, "empty_scissor.png");
    assert!(img.is_black());
}

// ---------------------------------------------------------------------------
// Abnormal inputs must fail explicitly.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "quad_to control point must be finite")]
fn panic_on_nan_quad_to() {
    let (mut vger, _, _) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    vger.quad_to([f32::NAN, 0.0], [1.0, 1.0]);
}

#[test]
#[should_panic(expected = "move_to point must be finite")]
fn panic_on_infinite_move_to() {
    let (mut vger, _, _) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    vger.move_to([f32::INFINITY, 0.0]);
}

#[test]
#[should_panic(expected = "stroke_segment width must be finite and non-negative")]
fn panic_on_negative_stroke_width() {
    let (mut vger, _, _) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    vger.stroke_segment([0.0, 0.0], [10.0, 10.0], -1.0, cyan);
}

#[test]
#[should_panic(expected = "fill_circle radius must be finite and non-negative")]
fn panic_on_nan_radius() {
    let (mut vger, _, _) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    vger.fill_circle([100.0, 100.0], f32::NAN, cyan);
}

#[test]
#[should_panic(expected = "scissor rect must be finite")]
fn panic_on_nan_scissor() {
    let (mut vger, _, _) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    vger.scissor(euclid::rect(f32::NAN, 0.0, 100.0, 100.0));
}

#[test]
#[should_panic(expected = "fill_blurred_rect blur_radius must be finite and positive")]
fn panic_on_zero_blur() {
    let (mut vger, _, _) = make_vger();
    vger.begin(512.0, 512.0, 1.0);
    let cyan = vger.color_paint(Color::CYAN);
    vger.fill_blurred_rect(euclid::rect(0.0, 0.0, 100.0, 100.0), 4.0, cyan, 0.0);
}
