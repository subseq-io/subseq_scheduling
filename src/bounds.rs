use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, PartialEq},
    collections::VecDeque,
    ops::Add,
};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash, Serialize, Deserialize)]
pub struct Attention(u8);

pub const MAX_ATTENTION: u8 = 100;
pub const DEFAULT_ATTENTION: u8 = 75;
pub const PARTIAL_ATTENTION: u8 = 50;
pub const MIN_ATTENTION: u8 = 25;
pub const NO_ATTENTION: u8 = 0;

impl Attention {
    pub fn new(attention: u8) -> Self {
        Attention(attention.min(MAX_ATTENTION))
    }

    pub fn value(&self) -> u8 {
        self.0
    }

    pub const NO_MULTITASKING: Self = Attention(MAX_ATTENTION);
    pub const DEFAULT_MULTITASKING: Self = Attention(DEFAULT_ATTENTION);
    pub const PARTIAL_MULTITASKING: Self = Attention(PARTIAL_ATTENTION);
    pub const FULL_MULTITASKING: Self = Attention(MIN_ATTENTION);
    pub const TRACKING: Self = Attention(NO_ATTENTION);

    pub fn can_multitask_with(&self, other: &Self) -> bool {
        self.0.saturating_add(other.0) <= 100
    }
}

impl Add for Attention {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Attention::new(self.0 + other.0)
    }
}

impl Default for Attention {
    fn default() -> Self {
        Attention::DEFAULT_MULTITASKING
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum Violation {
    /// The value must be equal to or above the given bound.
    Lower(f64),
    /// The value must be equal to or below the given bound.
    Upper(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Window {
    pub start: f64, // Inclusive
    pub end: f64,   // Exclusive
}

impl Eq for Window {}
impl std::hash::Hash for Window {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.start.to_bits().hash(state);
        self.end.to_bits().hash(state);
    }
}

impl Ord for Window {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start
            .partial_cmp(&other.start)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.end.partial_cmp(&other.end).unwrap_or(Ordering::Equal))
    }
}

impl Window {
    pub fn new(start: f64, end: f64) -> Self {
        Window { start, end }
    }

    pub fn overlap(&self, other: Window) -> bool {
        // Case 1: The range is entirely within the outer range.
        let case1 = other.start >= self.start && other.end <= self.end;
        // Case 2: The other overlaps with the lower bound at the end.
        let case2 = other.start < self.start && other.end > self.start;
        // Case 3: The other overlaps with the upper bound at the start.
        let case3 = other.start < self.end && other.end > self.end;
        // Case 4: The other encloses the outer other.
        let case4 = other.start < self.start && other.end > self.end;
        case1 || case2 || case3 || case4
    }

    pub fn add_offset(&self, offset: f64) -> Window {
        Window {
            start: self.start + offset,
            end: self.end + offset,
        }
    }

    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Bound {
    /// The range must start after the given value.
    Lower(f64),
    /// The range must end before or on the given value.
    Upper(f64),
}

impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Bound::Lower(a), Bound::Lower(b)) => a.partial_cmp(b),
            (Bound::Upper(a), Bound::Upper(b)) => a.partial_cmp(b),
            (Bound::Lower(a), Bound::Upper(b)) => a.partial_cmp(b),
            (Bound::Upper(a), Bound::Lower(b)) => a.partial_cmp(b),
        }
    }
}

impl Bound {
    pub fn violated(&self, range: Window) -> Option<Violation> {
        match self {
            Bound::Lower(expected) => {
                if range.start < *expected {
                    Some(Violation::Lower(*expected))
                } else {
                    None
                }
            }
            Bound::Upper(expected) => {
                if range.end > *expected {
                    Some(Violation::Upper(*expected))
                } else {
                    None
                }
            }
        }
    }
}

/// Repeated bound, block off this range ever nth repeat.
/// This is used for representing repeated constraints such as weekends.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepeatedBound {
    n: usize,
    bound: Window,
}

impl RepeatedBound {
    pub fn new(n: usize, bound: Window) -> Self {
        RepeatedBound { n, bound }
    }

    pub fn violated(&self, range: Window) -> Option<Window> {
        // Check if the range overlaps with the nth bound.
        // Get the nth bound before and after the range.
        let offsets = [-1, 0, 1];
        for offset in offsets.into_iter() {
            let offset_range = self.range(offset, range.start);
            if offset_range.overlap(range) {
                return Some(offset_range);
            }
        }
        None
    }

    pub fn range(&self, offset: isize, time: f64) -> Window {
        let segment_offset = time as isize / self.n as isize + offset;
        let segment_value = self.n as f64 * segment_offset as f64;
        self.bound.add_offset(segment_value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardConstraint(Bound);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Constraint {
    /// A hard constraint is a constraint that must be satisfied for the solution
    HardConstraint(HardConstraint),
    /// A block is a constraint that blocks off a range of time, such as weekends.
    /// During a block, events are paused and workers are not available.
    RepeatedBlock(RepeatedBound),
    Block(Window),
}

pub enum ViolatedConstriant {
    Hard(Violation),
    Block(Window),
}

impl Constraint {
    pub fn is_hard(&self) -> bool {
        match self {
            Constraint::HardConstraint { .. } => true,
            _ => false,
        }
    }

    pub fn evaluate(self, range: Window) -> Option<ViolatedConstriant> {
        match self {
            Constraint::HardConstraint(bound) => {
                let violation = bound.0.violated(range);
                violation.map(ViolatedConstriant::Hard)
            }
            Constraint::RepeatedBlock(repeated_bound) => {
                if let Some(violated_range) = repeated_bound.violated(range) {
                    Some(ViolatedConstriant::Block(violated_range))
                } else {
                    None
                }
            }
            Constraint::Block(block_range) => {
                if block_range.overlap(range) {
                    Some(ViolatedConstriant::Block(block_range))
                } else {
                    None
                }
            }
        }
    }
}

impl PartialOrd for Constraint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Constraint::HardConstraint(a), Constraint::HardConstraint(b)) => a.partial_cmp(b),
            (Constraint::RepeatedBlock(a), Constraint::RepeatedBlock(b)) => a.partial_cmp(b),
            (Constraint::Block(a), Constraint::Block(b)) => a.partial_cmp(b),
            (Constraint::HardConstraint(_), _) => Some(Ordering::Less),
            (_, Constraint::HardConstraint(_)) => Some(Ordering::Greater),
            (Constraint::RepeatedBlock(_), _) => Some(Ordering::Less),
            (_, Constraint::RepeatedBlock(_)) => Some(Ordering::Greater),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Constraints(Vec<Constraint>);

impl Default for Constraints {
    fn default() -> Self {
        Constraints::new()
    }
}

#[derive(Debug)]
struct RepeatedBoundState {
    offset: isize,
    bound: RepeatedBound,
    last_range: Window,
}

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct Consideration {
    pub window: Window,
    pub attention: Attention,
}

impl Consideration {
    pub fn new(window: Window, attention: Attention) -> Self {
        Self { window, attention }
    }
}

impl Eq for Consideration {}

impl From<Window> for Consideration {
    fn from(window: Window) -> Self {
        Consideration {
            window,
            attention: Attention::NO_MULTITASKING,
        }
    }
}

impl Ord for Consideration {
    fn cmp(&self, other: &Self) -> Ordering {
        self.window.cmp(&other.window)
    }
}

pub struct BlockIterator {
    repeated_blocks: Vec<RepeatedBoundState>,
    time: f64,
    next_blocks: VecDeque<Consideration>,
}

impl BlockIterator {
    pub fn new(constraints: &[Constraint], time: f64, considerations: Vec<Consideration>) -> Self {
        let mut blocks = vec![];
        let mut repeated_blocks = vec![];
        for block in considerations {
            blocks.push(block);
        }

        for block in constraints {
            match block {
                Constraint::Block(range) => blocks.push((*range).into()),
                Constraint::RepeatedBlock(repeated_bound) => {
                    let next_range = repeated_bound.range(-1, time);
                    blocks.push(next_range.into());

                    repeated_blocks.push(RepeatedBoundState {
                        offset: 0,
                        bound: *repeated_bound,
                        last_range: next_range,
                    });
                }
                _ => {}
            }
        }
        blocks.sort();

        BlockIterator {
            repeated_blocks,
            time,
            next_blocks: blocks.into(),
        }
    }

    fn next_repeated_blocks(&mut self) {
        let mut next_blocks: Vec<Consideration> = vec![];
        let first_block = self
            .repeated_blocks
            .first()
            .as_ref()
            .map(|block| block.last_range);
        for block in self.repeated_blocks.iter_mut() {
            if let Some(first_block) = first_block.as_ref() {
                if &block.last_range > first_block {
                    // Block is still in the future
                    continue;
                }
            }
            let next_range = block.last_range.add_offset(block.bound.n as f64);
            block.offset += 1;
            block.last_range = next_range;
            next_blocks.push(next_range.into());
        }
        self.next_blocks.extend(next_blocks);
        self.next_blocks.make_contiguous().sort();
    }
}

impl Iterator for BlockIterator {
    type Item = Consideration;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_repeated_blocks();
        if let Some(next_block) = self.next_blocks.pop_front() {
            self.time = next_block.window.end;
            Some(next_block)
        } else {
            None
        }
    }
}

impl Constraints {
    pub fn new() -> Self {
        Constraints(Vec::new())
    }

    pub fn add_block(mut self, block: Window) -> Self {
        self.0.push(Constraint::Block(block));
        self
    }

    pub fn add_repeated_block(mut self, bound: RepeatedBound) -> Self {
        self.0.push(Constraint::RepeatedBlock(bound));
        self
    }

    pub fn add_hard_bound(self, bound: Bound) -> Self {
        match bound {
            Bound::Lower(value) => self.lower_bound(value, Ordering::Greater),
            Bound::Upper(value) => self.upper_bound(value, Ordering::Less),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.0.iter()
    }

    fn lower_bound(mut self, lower_bound: f64, ord: Ordering) -> Self {
        let mut modified = false;
        for constraint in self.0.iter_mut() {
            if let Constraint::HardConstraint(HardConstraint(Bound::Lower(value))) = constraint {
                if (*value).partial_cmp(&lower_bound) == Some(ord) {
                    *value = lower_bound;
                }
                // We mark modified as true even if the value is the same, because we want to
                // there is only ever one lower bound.
                modified = true;
                break;
            }
        }
        if !modified {
            self.0
                .push(Constraint::HardConstraint(HardConstraint(Bound::Lower(
                    lower_bound,
                ))));
        }
        self
    }

    fn upper_bound(mut self, upper_bound: f64, ord: Ordering) -> Self {
        let mut modified = false;
        for constraint in self.0.iter_mut() {
            if let Constraint::HardConstraint(HardConstraint(Bound::Upper(value))) = constraint {
                if (*value).partial_cmp(&upper_bound) == Some(ord) {
                    *value += upper_bound;
                }
                modified = true;
                break;
            }
        }
        if !modified {
            self.0
                .push(Constraint::HardConstraint(HardConstraint(Bound::Upper(
                    upper_bound,
                ))));
        }
        self
    }

    /// Returns an iterator over all blocks that are active at the given time.
    pub fn block_iter(
        &self,
        start: f64,
        considerations: Vec<Consideration>,
    ) -> impl Iterator<Item = Consideration> {
        BlockIterator::new(&self.0, start, considerations)
    }

    /// Returns an iterator over all hard bounds that are active at the given time.
    pub fn hard_bounds(&self) -> Vec<Bound> {
        self.0
            .iter()
            .filter_map(|constraint| {
                if let Constraint::HardConstraint(HardConstraint(bound)) = constraint {
                    Some(*bound)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn hard_bound_fn(&self) -> impl Fn(Window) -> Vec<Violation> {
        let hard_constraints = self.hard_bounds();
        move |range| {
            let mut violations = vec![];
            for bound in hard_constraints.iter() {
                if let Some(violation) = bound.violated(range) {
                    violations.push(violation);
                }
            }
            violations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_add() {
        let attention = Attention::new(50);
        let attention2 = Attention::new(25);
        let attention3 = Attention::new(75);
        assert_eq!(attention + attention2, attention3);
    }

    #[test]
    fn test_attention_max() {
        let attention = Attention::new(100);
        let attention2 = Attention::new(50);
        assert_eq!(attention + attention2, attention);
    }

    #[test]
    fn test_attention_can_multitask() {
        let attention = Attention::new(50);
        let attention2 = Attention::new(25);
        assert!(attention.can_multitask_with(&attention2));

        let attention = Attention::new(75);
        let attention2 = Attention::new(25);
        assert!(attention.can_multitask_with(&attention2));

        let attention = Attention::new(75);
        let attention2 = Attention::new(50);
        assert!(!attention.can_multitask_with(&attention2));

        let attention = Attention::new(75);
        let attention2 = Attention::new(100);
        assert!(!attention.can_multitask_with(&attention2));
    }

    #[test]
    fn test_overlap_case1() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let other = Window {
            start: 3.0,
            end: 7.0,
        };
        assert!(window.overlap(other));
    }

    #[test]
    fn test_overlap_case2() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let other = Window {
            start: -5.0,
            end: 5.0,
        };
        assert!(window.overlap(other));
    }

    #[test]
    fn test_overlap_case3() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let other = Window {
            start: 5.0,
            end: 15.0,
        };
        assert!(window.overlap(other));
    }

    #[test]
    fn test_overlap_case4() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let other = Window {
            start: -5.0,
            end: 15.0,
        };
        assert!(window.overlap(other));
    }

    #[test]
    fn test_non_overlap() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let other = Window {
            start: 11.0,
            end: 20.0,
        };
        assert!(!window.overlap(other));
        let other2 = Window {
            start: -5.0,
            end: 0.0,
        };
        assert!(!window.overlap(other2));
    }

    #[test]
    fn test_exact_match() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let other = Window {
            start: 0.0,
            end: 10.0,
        };
        assert!(window.overlap(other));
    }

    #[test]
    fn test_repeated_bound_no_violation() {
        let rb = RepeatedBound {
            n: 7,
            bound: Window {
                start: 6.0,
                end: 8.0,
            },
        };
        let range = Window {
            start: 15.0,
            end: 16.0,
        };
        assert_eq!(rb.violated(range), None);
    }

    #[test]
    fn test_repeated_bound_violation() {
        let rb = RepeatedBound {
            n: 7,
            bound: Window {
                start: 6.0,
                end: 8.0,
            },
        };
        let range = Window {
            start: 7.0,
            end: 9.0,
        };
        assert!(rb.violated(range).is_some());
    }

    #[test]
    fn test_repeated_bound_boundary_range_computation() {
        let rb = RepeatedBound {
            n: 7,
            bound: Window {
                start: 6.0,
                end: 8.0,
            },
        };
        let offset_range = rb.range(1, 13.0);
        assert_eq!(
            offset_range,
            Window {
                start: 20.0,
                end: 22.0
            }
        );
        let offset_range = rb.range(0, 13.0);
        assert_eq!(
            offset_range,
            Window {
                start: 13.0,
                end: 15.0
            }
        );
        let offset_range = rb.range(-1, 13.0);
        assert_eq!(
            offset_range,
            Window {
                start: 6.0,
                end: 8.0,
            }
        );
    }

    #[test]
    fn test_ord_window() {
        let window = Window {
            start: 0.0,
            end: 10.0,
        };
        let window2 = Window {
            start: 5.0,
            end: 15.0,
        };
        let window3 = Window {
            start: 0.0,
            end: 15.0,
        };
        let window4 = Window {
            start: 0.0,
            end: 10.0,
        };
        assert!(window < window2);
        assert!(window2 > window);
        assert!(window < window3);
        assert!(window3 > window);
        assert!(window == window4);
    }

    #[test]
    fn test_single_fixed_block() {
        let considerations = vec![Window {
            start: 5.0,
            end: 10.0,
        }
        .into()];
        let constraints = vec![Constraint::Block(Window {
            start: 0.0,
            end: 5.0,
        })];
        let mut bi = BlockIterator::new(&constraints, 0.0, considerations);

        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 0.0,
                    end: 5.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 5.0,
                    end: 10.0
                }
                .into()
            )
        );
        assert_eq!(bi.next(), None);
    }

    #[test]
    fn test_single_repeated_block() {
        let considerations = vec![
            Window {
                start: 8.0,
                end: 15.0,
            }
            .into(),
            Window {
                start: 18.0,
                end: 32.0,
            }
            .into(),
        ];
        let constraints = vec![
            Constraint::RepeatedBlock(RepeatedBound {
                n: 7,
                bound: Window {
                    start: 6.0,
                    end: 8.0,
                },
            }),
            Constraint::Block(Window {
                start: 0.0,
                end: 5.0,
            }),
        ];
        let mut bi = BlockIterator::new(&constraints, 0.0, considerations);

        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: -1.0,
                    end: 1.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 0.0,
                    end: 5.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 6.0,
                    end: 8.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 8.0,
                    end: 15.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 13.0,
                    end: 15.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 18.0,
                    end: 32.0
                }
                .into()
            )
        );
        assert_eq!(
            bi.next(),
            Some(
                Window {
                    start: 20.0,
                    end: 22.0
                }
                .into()
            )
        );
    }
}
