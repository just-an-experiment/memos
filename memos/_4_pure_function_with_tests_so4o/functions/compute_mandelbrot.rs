/// Computes whether a given complex number (defined by its real and imaginary parts)
/// is in the Mandelbrot set, given a maximum number of iterations.
///
/// # Parameters
/// - `real`: The real part of the complex number.
/// - `imaginary`: The imaginary part of the complex number.
/// - `max_iterations`: The maximum number of iterations to determine if the point is in the set.
///
/// # Returns
/// - `bool`: Returns `true` if the complex number is in the Mandelbrot set within the given number of iterations, otherwise `false`.
fn compute_mandelbrot(real: f64, imaginary: f64, max_iterations: u32) -> bool {
    let c = num::Complex::new(real, imaginary);
    let mut z = num::Complex::new(0.0, 0.0);
    
    for _ in 0..max_iterations {
        if z.norm_sqr() > 4.0 {
            return false;
        }
        z = z * z + c;
    }
    true
}

fn test_compute_mandelbrot() {
    assert_eq!(compute_mandelbrot(0.0, 0.0, 1000), true, "Origin is in the Mandelbrot set");
    assert_eq!(compute_mandelbrot(2.0, 2.0, 1000), false, "Point (2,2) is outside the Mandelbrot set");
    assert_eq!(compute_mandelbrot(-1.0, 0.0, 1000), true, "Point (-1,0) is inside the Mandelbrot set");
    assert_eq!(compute_mandelbrot(0.5, 0.5, 1000), false, "Point (0.5,0.5) is outside the Mandelbrot set");
    assert_eq!(compute_mandelbrot(-1.75, 0.0, 10), false, "Point (-1.75,0) may diverge with too few iterations");
    assert_eq!(compute_mandelbrot(-0.75, 0.0, 1000), true, "Point (-0.75,0) is inside the Mandelbrot set");
    assert_eq!(compute_mandelbrot(0.3, -0.1, 1000), true, "Point (0.3,-0.1) is inside a bulb of the Mandelbrot set");
    assert_eq!(compute_mandelbrot(-2.0, 0.0, 1000), false, "Point (-2,0) is outside the Mandelbrot set");
}

pub use test_compute_mandelbrot as tests;
pub use compute_mandelbrot as default;