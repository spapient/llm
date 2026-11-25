use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measure(x: &[f32], positions: &mut [f32], out: &mut [f32], convert: bool) -> (f64, usize) {
    let iterations = 20_000;
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for position in 0..iterations {
        positions[0] = position as f32;
        if convert {
            let converted: Vec<usize> = positions.iter().map(|&value| value as usize).collect();
            nn::rope_f32::forward_batch(
                black_box(&converted),
                tensor::View::one(black_box(x)),
                black_box(&mut *out),
                2,
            )
            .unwrap();
        } else {
            nn::rope_f32::forward_batch_from_f32(
                black_box(&*positions),
                tensor::View::one(black_box(x)),
                black_box(&mut *out),
                2,
            )
            .unwrap();
        }
    }
    (
        start.elapsed().as_nanos() as f64 / f64::from(iterations),
        ALLOCATIONS.load(Ordering::Relaxed),
    )
}

fn main() {
    let x: Vec<f32> = (0..64).map(|idx| idx as f32 * 0.125).collect();
    let mut positions = [0.0f32; 32];
    let mut out = vec![0.0; x.len()];
    measure(&x, &mut positions, &mut out, true);
    measure(&x, &mut positions, &mut out, false);
    let mut baseline = Vec::new();
    let mut optimized = Vec::new();
    let mut converted_allocations = 0;
    let mut direct_allocations = 0;
    for _ in 0..11 {
        let (time, allocations) = measure(&x, &mut positions, &mut out, true);
        baseline.push(time);
        converted_allocations += allocations;
        let (time, allocations) = measure(&x, &mut positions, &mut out, false);
        optimized.push(time);
        direct_allocations += allocations;
    }
    baseline.sort_by(f64::total_cmp);
    optimized.sort_by(f64::total_cmp);
    println!("converted_median_ns={:.3} optimized_median_ns={:.3}", baseline[5], optimized[5]);
    println!("converted_allocations={converted_allocations} direct_allocations={direct_allocations}");
    println!("converted_ns={baseline:?}");
    println!("direct_ns={optimized:?}");
}
