#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int l = 0; l < m; l += batch) {
        int r = std::min(l + batch, m);
        printf("l = %d, r = %d\n", l, r);

        double Iy[(r-l) * k];
        for(int i = l; i < r; ++ i) {
            for(int kk = 0; kk < k; ++ kk) {
                Iy[(i-l)*k + kk] = (y[i] == kk) ? 1.0 : 0;
            }
        }

        float Z[(r-l) * k];
        for(int i = l; i < r; ++ i) {
            double sum = 0;
            for(int kk = 0; kk < k; ++ kk) {
                Z[(i-l)*k+kk] = 0;
                for(int j = 0; j < n; ++ j) {
//                    printf("%.2f %.2f\n", X[i*n+j], theta[j*k+kk]);
                    Z[(i-l)*k+kk] += std::exp(X[i*n+j] * theta[j*k+kk]);
                }
                sum += Z[(i-l)*k+kk];
            }
            // normalize - Iy
//            printf("sum = %.2f\n", sum);
            for(int kk = 0; kk < k; ++ kk) {
                Z[(i-l)*k+kk] /= sum;
                Z[(i-l)*k+kk] -= Iy[(i-l)*k+kk];
            }
        }

        for(int j = 0; j < n; ++ j) {
            for(int kk = 0; kk < k; ++ kk) {
                double tmp = 0;
                for(int i = l; i < r; ++ i) {
                    tmp += X[i*n+j] * Z[(i-l)*k+kk];
                }
                theta[j*k+kk] -= lr * tmp / (r-l);
            }
        }
    }



    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
