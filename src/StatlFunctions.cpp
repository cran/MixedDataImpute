#include "StatFunctions.h"
#include <math.h>
#include <cfloat>
#include <stdexcept>
#include <sstream>

// Written by John D. Cook, http://www.johndcook.com

//-----------------------------------------------------------------------------

// Calculate log(1 + x), preventing loss of precision for small values of x.
// The input x must be larger than -1 so that log(1 + x) is real.
double LogOnePlusX(double x)
{
  /*
    if (x <= -1.0)
    {
        std::stringstream os;
        os << "Invalid input argument (" << x << "); must be greater than -1.0";
		throw std::invalid_argument(os.str());
    }
*/
	if (fabs(x) > 0.375)
    {
        // x is sufficiently large that the obvious evaluation is OK
        return log(1.0 + x);
    }

	// For smaller arguments we use a rational approximation
	// to the function log(1+x) to avoid the loss of precision
	// that would occur if we simply added 1 to x then took the log.

    const double p1 =  -0.129418923021993e+01;
    const double p2 =   0.405303492862024e+00;
    const double p3 =  -0.178874546012214e-01;
    const double q1 =  -0.162752256355323e+01;
    const double q2 =   0.747811014037616e+00;
    const double q3 =  -0.845104217945565e-01;
    double t, t2, w;

    t = x/(x + 2.0);
    t2 = t*t;
    w = (((p3*t2 + p2)*t2 + p1)*t2 + 1.0)/(((q3*t2 + q2)*t2 + q1)*t2 + 1.0);
    return 2.0*t*w;
}

//-----------------------------------------------------------------------------

// Calculate exp(x) - 1.
// The most direct method is inaccurate for very small arguments.
double ExpMinusOne(double x)
{
    const double p1 =  0.914041914819518e-09;
    const double p2 =  0.238082361044469e-01;
    const double q1 = -0.499999999085958e+00;
    const double q2 =  0.107141568980644e+00;
    const double q3 = -0.119041179760821e-01;
    const double q4 =  0.595130811860248e-03;

	double rexp = 0.0;

	// Use rational approximation for small arguments.
    if( fabs(x) < 0.15 )
    {
        rexp = x*(((p2*x + p1)*x + 1.0)/((((q4*x + q3)*x + q2)*x + q1)*x + 1.0));
        return rexp;
    }

	// For large negative arguments, direct calculation is OK.
    double w = exp(x);
    if( x <= -0.15 )
    {
        rexp = w - 1.0;
        return rexp;
    }

	// The following expression is algebraically equal to exp(x) - 1.
	// The advantage in finite precision arithmetic is that
	// it avoids subtracting nearly equal numbers.
    rexp = w * ( 0.5 + ( 0.5 - 1.0/w ));
    return rexp;
}

//-----------------------------------------------------------------------------

// logit(p) = log(p/(1-p))
// The argument p must be greater than 0 and less than 1.
double Logit(double p)
{
    if( (p <= 0.0) || (p >= 1.0) )
    {
        std::stringstream os;
        os << "argument (" << p << ") must be greater than 0 and less than 1.";
		throw std::invalid_argument( os.str() );
    }

    static const double smallCutOff = 0.25;

    double retval;

    if (p < smallCutOff)
    {
        // Avoid calculating 1-p since the lower bits of p would be lost.
        retval = log(p) - LogOnePlusX(-p);
    }
    else
    {
		// The argument p is large enough that direct calculation is OK.
        retval = log(p/(1-p));
    }
    return retval;
}

//-----------------------------------------------------------------------------

// The inverse of the Logit function. Return exp(x)/(1 + exp(x)).
// Avoid overflow and underflow for extreme inputs.
double LogitInverse(double x)
{
    static const double X_MAX = -log(DBL_EPSILON);
    static const double X_MIN =  log(DBL_MIN);
    double retval;

    if (x > X_MAX)
    {
        // For large arguments x, logit(x) equals 1 to double precision.
        retval = 1.0;  // avoids overflow of calculating e^x for large x
    }
    else if (x < X_MIN)
    {
        // logit(x) is approximately e^x for x very negative
        // and so logit would underflow when e^x underflows
        retval = 0.0;
    }
    else
    {
        // Direct calculation is safe in this range.
		// Save value to avoid two calls to e^x
        double t = exp(x);
        retval = t/(1+t);
    }

    return retval;
}

//-----------------------------------------------------------------------------

// The natural logorithm of the logit inverse function.
// Return log( exp(x)/(1 + exp(x)) )
double LogLogitInverse(double x)
{
    // log( exp(x)/(1 + exp(x) ) = x - log(1 + exp(x)).
    // For x < -30, x - log(1 + exp(x)) = x to machine precision
    // since the log term is extremely small relative to x.
    if (x < -30)
        return x;

    // The obvious implementation is OK in the middle range.
    if (x < 12)
        return log(LogitInverse(x));

	// Set y = exp(x). Then x - log(1 + exp(x)) = log(y) - log(y + 1).
	// Expand in Taylor series around y.
	// log(y) - log(y+1) = - 1/y - 1/y^2 + O(1/y^3).
	// Since x >= 12, 1/y^3 is extremely small.
    double one_over_y = exp(-x);
    return -(1.0 - 0.5*one_over_y)*one_over_y;
}

//-----------------------------------------------------------------------------

// Compute LogitInverse(x) - LogitInverse(y) accurately,
// especially for approximately equal values of x and y
// and for large values of x and y.
double LogitInverseDifference(double x, double y)
{
    static const double CLOSE_CUTOFF = 0.25;
    static const double LOG_DBL_MAX  = log(DBL_MAX);
    static const double LOG_DBL_MIN  = log(DBL_MIN);

    if (fabs(x-y) < CLOSE_CUTOFF)
    {
        if (x > LOG_DBL_MAX || x < LOG_DBL_MIN)
        {
            // For numbers this large in absolute value, the difference
            // of their logitInverse values is 0 to machine precision.
            // Return 0 and avoid overflow.
            return 0.0;
        }
        else
        {
            // Use expMinusOne to avoid cancellation in exp(x-y) - 1.
            // This cannot overflow since |x-y| < CLOSE_CUTOFF.
            // Other exponents safe due to range of x (and thus y).
            return ExpMinusOne(x-y)/((exp(x) + 1.0)*(exp(-y) + 1.0));
        }
    }
    else
    {
        bool x_positive = (x > 0.0);
        bool y_positive = (y > 0.0);

        if (x_positive && y_positive)
        {
            // logitInverse(x) - logitInverse(y) == logitInverse(-y) - logitInverse(-x)
            // swap (x, y) with (-y, -x) so that both arguments are negative
            double temp = x; x = -y; y = -temp;

            // might underflow, but cannot overflow since arguments are negative
            double a = exp(x), b = exp(y);

            // The following subtraction won't lose precision since |x-y| > SMALL_CUTOFF.
            return (a - b)/((1.0 + a)*(1.0 + b));
        }
        else if (!x_positive && !y_positive)
        {
            // See comments for case x > 0 and y > 0.
            double a = exp(x), b = exp(y);
            return (a - b)/((1.0 + a)*(1.0 + b));
        }
        else if (x_positive && !y_positive)
        {
            return (1.0 - exp(y-x))/((1.0 + exp(-x))*(1.0 + exp(y)));
        }
        else
        {
            return (exp(x-y) - 1.0)/((1.0 + exp(-y))*(1.0 + exp(x)));
        }
    }
}

//-----------------------------------------------------------------------------

// return log(1 + exp(x)), preventing cancellation and overflow */
double LogOnePlusExpX(double x)
{
    static const double LOG_DBL_EPSILON = log(DBL_EPSILON);
    static const double LOG_ONE_QUARTER = log(0.25);

    if (x > -LOG_DBL_EPSILON)
    {
        // log(exp(x) + 1) == x to machine precision
        return x;
    }
    else if (x > LOG_ONE_QUARTER)
    {
        return log( 1.0 + exp(x) );
    }
    else
    {
        // Prevent loss of precision that would result from adding small argument to 1.
        return LogOnePlusX( exp(x) );
    }
}
//-----------------------------------------------------------------------------

// Calculate log( -log(1 - p) ) avoiding problems for small values of p.
// Input p must be strictly between 0 and 1
double ComplementaryLogLog(double p)
{

    if (p <= 0.0 || p >= 1.0)
	{
        std::stringstream os;
        os << "Invalid input argument (" << p << "); must be greater than 0 and less than 1.";
		throw std::invalid_argument(os.str());
	}
    return log( -LogOnePlusX(-p) );
}

//-----------------------------------------------------------------------------

// Compute 1.0 - exp(-exp(x)), avoiding numerical problems with |x| large
double ComplementaryLogLogInverse(double x)
{
    // In theory, we could directly evaluate 1.0 - exp(-exp(x)).
    // However, if x is too large, the inner exp overflows even though the
    // final result is near 1.  Also, if x is too negative,
    // exp(-exp(x)) is close to 1 and precision is lost in
    // subtracting this amount from 1.

    if (x > 3.584730797999763)
    {
        // Prevent overflow in itermediate result.
        return 1.0; // Exact value equals 1 to machine precision.
    }
    else if (x > -18.420680743952365472)
    {
        // At the cutoff value, the subtraction result is accurate to single precision
        return 1.0 - exp(-exp(x));
    }
    else
    {
        // This approximation is better than the exact function for arguments this small.
        // Let y = exp(x).  Then f(x) = 1 - exp(-y) = y - y^2/2 - ...
        // and so the absolute error in approximating 1 - exp(y) with -y is roughly y^2/2
        // and the relative error is roughly y/2.
        // If x < -18.42... then y < 10^-8 and so the approximation is good to at least single precision.
        return exp(x);
    }
}
