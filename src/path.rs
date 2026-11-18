#![allow(dead_code)]

use crate::defs::*;

pub struct Interval {
    pub a: f32,
    pub b: f32,
}

pub struct PathSegment {
    pub cvs: [LocalPoint; 3],
    pub next: Option<usize>,
    previous: Option<usize>,
}

impl PathSegment {
    pub fn new(a: LocalPoint, b: LocalPoint, c: LocalPoint) -> Self {
        Self {
            cvs: [a, b, c],
            next: None,
            previous: None,
        }
    }

    pub fn y_interval(&self) -> Interval {
        Interval {
            // Fatten the interval slightly to prevent artifacts by
            // slightly missing a curve in a band.
            a: self.cvs[0].y.min(self.cvs[1].y).min(self.cvs[2].y) - 1.0,
            b: self.cvs[0].y.max(self.cvs[1].y).max(self.cvs[2].y) + 1.0,
        }
    }
}

#[derive(PartialEq, PartialOrd)]
struct PathScannerNode {
    coord: f32,
    seg: usize,
    end: bool,
}

pub struct PathScanner {
    pub segments: Vec<PathSegment>,
    /// Indices in `segments` where new subpaths begin (each `move_to`
    /// after the first). Every subpath is closed independently so
    /// disjoint subpaths never get connected by a synthetic segment.
    subpath_starts: Vec<usize>,
    nodes: Vec<PathScannerNode>,
    index: usize,
    pub interval: Interval,
    pub first: Option<usize>,
}

impl PathScanner {
    pub fn new() -> Self {
        Self {
            segments: vec![],
            subpath_starts: vec![],
            nodes: vec![],
            index: 0,
            interval: Interval { a: 0.0, b: 0.0 },
            first: None,
        }
    }

    /// Marks the start of a new subpath at the current segment count.
    /// Called when the pen is moved after segments have been emitted.
    pub fn start_subpath(&mut self) {
        let n = self.segments.len();
        let last = self.subpath_starts.last().copied().unwrap_or(0);
        if n > last {
            self.subpath_starts.push(n);
        }
    }

    /// Clears all segments and subpath breaks.
    pub fn reset(&mut self) {
        self.segments.clear();
        self.subpath_starts.clear();
    }

    pub fn init(&mut self) {
        // Close each subpath independently.
        let mut ranges: Vec<(usize, usize)> = vec![];
        let mut prev = 0;
        for &b in &self.subpath_starts {
            if b > prev && b <= self.segments.len() {
                ranges.push((prev, b));
                prev = b;
            }
        }
        if self.segments.len() > prev {
            ranges.push((prev, self.segments.len()));
        }

        for (a, b) in ranges {
            let start = self.segments[a].cvs[0];
            let end = self.segments[b - 1].cvs[2];
            if start != end {
                self.segments.push(PathSegment {
                    cvs: [end, start.lerp(end, 0.5), start],
                    next: None,
                    previous: None,
                })
            }
        }

        self.nodes.clear();
        self.index = 0;

        for i in 0..self.segments.len() {
            let y_interval = self.segments[i].y_interval();
            self.nodes.push(PathScannerNode {
                coord: y_interval.a,
                seg: i,
                end: false,
            });
            self.nodes.push(PathScannerNode {
                coord: y_interval.b,
                seg: i,
                end: true,
            });
        }

        self.nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    pub fn begin(&mut self, cvs: &[LocalPoint]) {
        self.reset();

        let mut i = 0;
        while i + 2 < cvs.len() {
            self.segments.push(PathSegment {
                cvs: [cvs[i], cvs[i + 1], cvs[i + 2]],
                next: None,
                previous: None,
            });
            i += 2;
        }

        self.init();
    }

    pub fn next(&mut self) -> bool {
        // Empty (or exhausted) scan: nothing to do. This can happen
        // when fill() is called with no path segments.
        if self.index >= self.nodes.len() {
            return false;
        }

        let y = self.nodes[self.index].coord;
        self.interval.a = y;
        let n = self.nodes.len();

        while self.index < n && self.nodes[self.index].coord == y {
            let node = &self.nodes[self.index];
            assert!(node.seg < self.segments.len());

            if node.end {
                if let Some(prev) = self.segments[node.seg].previous {
                    self.segments[prev].next = self.segments[node.seg].next;
                }
                if let Some(next) = self.segments[node.seg].next {
                    self.segments[next].previous = self.segments[node.seg].previous;
                }
                if self.first == Some(node.seg) {
                    self.first = self.segments[node.seg].next
                }
                self.segments[node.seg].next = None;
                self.segments[node.seg].previous = None;
            } else {
                self.segments[node.seg].next = self.first;
                if let Some(first) = self.first {
                    self.segments[first].previous = Some(node.seg);
                }
                self.first = Some(node.seg);
            }

            self.index += 1;
        }

        if self.index < n {
            self.interval.b = self.nodes[self.index].coord
        }

        self.index < n
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_path_scanner() {
        let mut scan = PathScanner::new();

        let cvs = vec![
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
            [-1.0, 1.0].into(),
            [-1.0, 0.0].into(),
            [-1.0, -1.0].into(),
            [0.0, -1.0].into(),
            [1.0, -1.0].into(),
            [1.0, 0.0].into(),
        ];

        scan.begin(&cvs);

        assert_eq!(scan.segments.len(), 4);

        while scan.next() {
            print!(
                "interval {:?} {:?} active: ",
                scan.interval.a, scan.interval.b
            );

            let mut index = scan.first;
            while let Some(i) = index {
                print!("{:?} ", i);
                index = scan.segments[i].next;
            }

            println!();
        }
    }

    #[test]
    fn test_empty_scanner() {
        // fill() with no segments must not panic or produce bands.
        let mut scan = PathScanner::new();
        scan.init();
        assert!(!scan.next());

        let mut scan = PathScanner::new();
        scan.begin(&[]);
        assert!(!scan.next());
    }

    #[test]
    fn test_subpaths_closed_independently() {
        // Two disjoint squares, far apart in y. Each subpath must be
        // closed independently; no synthetic segment may connect them.
        let mut scan = PathScanner::new();

        // Square 1: (0, 0) - (10, 10).
        scan.segments.push(PathSegment::new(
            [0.0, 0.0].into(),
            [5.0, 0.0].into(),
            [10.0, 0.0].into(),
        ));
        scan.segments.push(PathSegment::new(
            [10.0, 0.0].into(),
            [10.0, 10.0].into(),
            [0.0, 10.0].into(),
        ));

        scan.start_subpath();

        // Square 2: (0, 100) - (10, 110).
        scan.segments.push(PathSegment::new(
            [0.0, 100.0].into(),
            [5.0, 100.0].into(),
            [10.0, 100.0].into(),
        ));
        scan.segments.push(PathSegment::new(
            [10.0, 100.0].into(),
            [10.0, 110.0].into(),
            [0.0, 110.0].into(),
        ));

        scan.init();

        // Two closing segments, one per subpath.
        assert_eq!(scan.segments.len(), 6);
        let close1 = &scan.segments[4].cvs;
        assert_eq!(close1[0], LocalPoint::new(0.0, 10.0));
        assert_eq!(close1[2], LocalPoint::new(0.0, 0.0));
        let close2 = &scan.segments[5].cvs;
        assert_eq!(close2[0], LocalPoint::new(0.0, 110.0));
        assert_eq!(close2[2], LocalPoint::new(0.0, 100.0));

        // The scan must yield an empty band in the gap between the
        // two subpaths (no active segments), which fill() will skip.
        let mut saw_empty_band = false;
        let mut band_count = 0;
        while scan.next() {
            band_count += 1;
            if scan.first.is_none() {
                saw_empty_band = true;
                // The empty band spans the gap between the squares.
                assert!(scan.interval.a >= 9.0 && scan.interval.b <= 101.0);
            }
        }
        assert!(saw_empty_band);
        assert!(band_count >= 3);
    }
}
