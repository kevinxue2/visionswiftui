#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits.h>
#include <float.h>
#include <iostream>
#include "clapack.h"  //matlab
#include "opencv2/core.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRU
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0
/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */
#define M_1_2_PI 1.57079632679489661923
#define M_1_4_PI 0.785398163

#define M_3_4_PI 2.35619449

#define M_1_8_PI 0.392699081
#define M_3_8_PI 1.178097245
#define M_5_8_PI 1.963495408
#define M_7_8_PI 2.748893572
#define M_4_9_PI 1.396263401595464  //80��
#define M_1_9_PI  0.34906585  //20��
#define M_1_10_PI 0.314159265358979323846   //18��
#define M_1_12_PI 0.261799387   //15��
#define M_1_15_PI 0.20943951    //12��
#define M_1_18_PI 0.174532925   //10��
/** 3/2 pi */
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI  6.28318530718
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

struct point2i //(or pixel).
{
	int x,y;
};

struct point2d
{
	double x,y;
};

struct point1d1i
{
	double data;
	int cnt;
};

struct point3d
{
	double x,y;
	double r;
};

struct point3i
{
	int x,y;
	int z;
};

struct point2d1i
{
	double x,y;
	int z;
};

struct  point5d
{
	double x,y;
	double a,b;
	double phi;
};

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
  double x1,y1,x2,y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x,y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx,dy;        /* (dx,dy) is vector oriented as the line segment,dx = cos(theta), dy = sin(theta) */
  int   polarity;     /* if the arc direction is the same as the edge direction, polarity = 1, else if opposite ,polarity = -1.*/
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
};

typedef struct
{
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  double ys,ye;  /* start and end Y values of current 'column' */
  int x,y;       /* coordinates of currently explored pixel */
} rect_iter;

typedef struct image_double_s
{
  double * data;
  int xsize,ysize;
} * image_double;


//==================================================================================================
//=============================miscellaneous functions==============================================
inline double min(double v1,double v2);

inline double max(double v1,double v2);

/** Compare doubles by relative error.

    The resulting rounding error after floating point computations
    depend on the specific operations done. The same number computed by
    different algorithms could present different rounding errors. For a
    useful comparison, an estimation of the relative rounding error
    should be considered and compared to a factor times EPS. The factor
    should be related to the cumulated rounding error in the chain of
    computation. Here, as a simplification, a fixed factor is used.
 */
int double_equal(double a, double b);

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
double angle_diff(double a, double b);

/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
double angle_diff_signed(double a, double b);

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
void error(char const * msg);

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */
double dist(double x1, double y1, double x2, double y2);

double dotProduct(point2d vec1, point2d vec2);

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
void rect_copy(struct rect * in, struct rect * out);//in is the src, out is the dst


/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the smaller
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
double inter_low(double x, double x1, double y1, double x2, double y2);

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the larger
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
double inter_hi(double x, double x1, double y1, double x2, double y2);

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
void ri_del(rect_iter * iter);

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

    See details in \ref rect_iter
 */
int ri_end(rect_iter * i);

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

    See details in \ref rect_iter
 */
void ri_inc(rect_iter * i);

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
rect_iter * ri_ini(struct rect * r);

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
void free_image_double(image_double i);

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
image_double new_image_double(int xsize, int ysize);

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
image_double new_image_double_ptr( int xsize, int ysize, double * data );

//=================================================================================================================
//===========================================LSD functions=========================================================
/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402    //ln10
#endif /* !M_LN10 */

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED    1

//���ڹ���Բ�������ر�Ǽ��ԣ�����ݶȵķ���ͻ��ķ���ָ��һ�£���ΪSAME_POLE,����ΪOPP_POLE,�ñ�ǳ�ʼ��Ϊ0
#define NOTDEF_POL 0
#define SAME_POL 1
#define OPP_POL  -1
/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist
{
  int x,y;
  struct coorlist * next;
};
typedef struct ntuple_list_s
{
  int size;
  int max_size;
  int dim;
  double * values;
} * ntuple_list;

/*----------------------------------------------------------------------------*/
/** Free memory used in n-tuple 'in'.
 */
static void free_ntuple_list(ntuple_list in);

/*----------------------------------------------------------------------------*/
/** Create an n-tuple list and allocate memory for one element.
    @param dim the dimension (n) of the n-tuple.
 */
static ntuple_list new_ntuple_list(int dim);

/*----------------------------------------------------------------------------*/
/** Enlarge the allocated memory of an n-tuple list.
 */
static void enlarge_ntuple_list(ntuple_list n_tuple);

/*----------------------------------------------------------------------------*/
/** Add a 7-tuple to an n-tuple list.
 */
static void add_7tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7 );

/*----------------------------------------------------------------------------*/
/** Add a 8-tuple to an n-tuple list.
 */
static void add_8tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7, int v8);

/** char image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize;
} * image_char;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
static void free_image_char(image_char i);

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
static image_char new_image_char(unsigned int xsize, unsigned int ysize);

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_char new_image_char_ini( unsigned int xsize, unsigned int ysize,
                                      unsigned char fill_value );

/*----------------------------------------------------------------------------*/
/** int image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_int_s
{
  int * data;
  unsigned int xsize,ysize;
} * image_int;

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
static image_int new_image_int(unsigned int xsize, unsigned int ysize);

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_int new_image_int_ini( unsigned int xsize, unsigned int ysize,
                                    int fill_value );

/** Compute a Gaussian kernel of length 'kernel->dim',
    standard deviation 'sigma', and centered at value 'mean'.

    For example, if mean=0.5, the Gaussian will be centered
    in the middle point2i between values 'kernel->values[0]'
    and 'kernel->values[1]'.
 */
static void gaussian_kernel(ntuple_list kernel, double sigma, double mean);

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

    For example, scale=0.8 will give a result at 80% of the original size.

    The image is convolved with a Gaussian kernel
    @f[
        G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
    @f]
    before the sub-sampling to prevent aliasing.

    The standard deviation sigma given by:
    -  sigma = sigma_scale / scale,   if scale <  1.0
    -  sigma = sigma_scale,           if scale >= 1.0

    To be able to sub-sample at non-integer steps, some interpolation
    is needed. In this implementation, the interpolation is done by
    the Gaussian kernel, so both operations (filtering and sampling)
    are done at the same time. The Gaussian kernel is computed
    centered on the coordinates of the required sample. In this way,
    when applied, it gives directly the result of convolving the image
    with the kernel and interpolated to that particular position.

    A fast algorithm is done using the separability of the Gaussian
    kernel. Applying the 2D Gaussian kernel is equivalent to applying
    first a horizontal 1D Gaussian kernel and then a vertical 1D
    Gaussian kernel (or the other way round). The reason is that
    @f[
        G(x,y) = G(x) * G(y)
    @f]
    where
    @f[
        G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
    @f]
    The algorithm first applies a combined Gaussian kernel and sampling
    in the x axis, and then the combined Gaussian kernel and sampling
    in the y axis.
 */
static image_double gaussian_sampler( image_double in, double scale,
                                      double sigma_scale );


/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point2i.

    The result is:
    - an image_double with the angle at each pixel, or NOTDEF if not defined.
    - the image_double 'modgrad' (a point2ier is passed as argument)
      with the gradient magnitude at each point2i.
    - a list of pixels 'list_p' roughly ordered by decreasing
      gradient magnitude. (The order is made by classifying point2is
      into bins by gradient magnitude. The parameters 'n_bins' and
      'max_grad' specify the number of bins and the gradient modulus
      at the highest bin. The pixels in the list would be in
      decreasing gradient magnitude, up to a precision of the size of
      the bins.)
    - a point2ier 'mem_p' to the memory used by 'list_p' to be able to
      free the memory when it is not used anymore.
 */
//����һ���ݶȽǶ�˳ʱ����ת90����align�Ƕ�ͼangles������ݶȽǶ���(gx,gy)->(-gy,gx)��
//���ݶȵ�ģ��ͼmodgrad,Ȼ����n_bins����α���򷵻�������ͷָ��list_p,������������
static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p,
                              image_double * modgrad, unsigned int n_bins );

/*----------------------------------------------------------------------------*/
/** Is point2i (x,y) aligned to angle theta, up to precision 'prec'?
 */
static int isaligned( int x, int y, image_double angles, double theta,
                      double prec );


/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using the Lanczos approximation.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
      \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                  (x+5.5)^{x+0.5} e^{-(x+5.5)}
    @f]
    so
    @f[
      \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                      + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
    @f]
    and
      q0 = 75122.6331530,
      q1 = 80916.6278952,
      q2 = 36308.2951477,
      q3 = 8687.24529705,
      q4 = 1168.92649479,
      q5 = 83.8676043424,
      q6 = 2.50662827511.
 */
static double log_gamma_lanczos(double x);

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using Windschitl method.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
        \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                    \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
    @f]
    so
    @f[
        \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                      + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
    @f]
    This formula is a good approximation when x > 15.
 */
static double log_gamma_windschitl(double x);

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x. When x>15 use log_gamma_windschitl(),
    otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/** Size of the table to store already computed inverse values.
 */
#define TABSIZE 100000

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}
    @f]

    The value -log10(NFA) is equivalent but more intuitive than NFA:
    - -1 corresponds to 10 mean false alarms
    -  0 corresponds to 1 mean false alarm
    -  1 corresponds to 0.1 mean false alarms
    -  2 corresponds to 0.01 mean false alarms
    -  ...

    Used this way, the bigger the value, better the detection,
    and a logarithmic scale is used.

    @param n,k,p binomial parameters.
    @param logNT logarithm of Number of Tests

    The computation is based in the gamma function by the following
    relation:
    @f[
        \left(\begin{array}{c}n\\k\end{array}\right)
        = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
    @f]
    We use efficient algorithms to compute the logarithm of
    the gamma function.

    To make the computation faster, not all the sum is computed, part
    of the terms are neglected based on a bound to the error obtained
    (an error of 10% in the result is accepted).
 */
static double nfa(int n, int k, double p, double logNT);

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
static double rect_nfa(struct rect * rec, image_double angles, double logNT);

/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.

    The following is the region inertia matrix A:
    @f[

        A = \left(\begin{array}{cc}
                                    Ixx & Ixy \\
                                    Ixy & Iyy \\
             \end{array}\right)

    @f]
    where

      Ixx =   sum_i G(i).(y_i - cx)^2

      Iyy =   sum_i G(i).(x_i - cy)^2

      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)

    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i and y_i are the coordinates of pixel i.
    - cx and cy are the coordinates of the center of th region.

    lambda1 and lambda2 are the eigenvalues of matrix A,
    with lambda1 >= lambda2. They are found by solving the
    characteristic polynomial:

      det( lambda I - A) = 0

    that gives:

      lambda1 = ( Ixx + Iyy + sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

      lambda2 = ( Ixx + Iyy - sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

    To get the line segment direction we want to get the angle the
    eigenvector associated to the smallest eigenvalue. We have
    to solve for a,b in:

      a.Ixx + b.Ixy = a.lambda2

      a.Ixy + b.Iyy = b.lambda2

    We want the angle theta = atan(b/a). It can be computed with
    any of the two equations:

      theta = atan( (lambda2-Ixx) / Ixy )

    or

      theta = atan( Ixy / (lambda2-Iyy) )

    When |Ixx| > |Iyy| we use the first, otherwise the second (just to
    get better numeric precision).
 */
static double get_theta( point2i * reg, int reg_size, double x, double y,
                         image_double modgrad, double reg_angle, double prec );

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of point2is.
 */
static void region2rect( point2i * reg, int reg_size,
						image_double modgrad, double reg_angle,
                         double prec, double p, struct rect * rec );

//�������ĺͽǶ��Ѿ�������ˣ����ֻ���о��ν��ơ���region2rect���⻹���������ĺͽǶȼ��㡣
static void region2rect2(point2i * reg, int reg_size,double reg_center_x,double reg_center_y,
					double reg_theta,double prec, double p, struct rect * rec );

/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point2i (x,y).
 */
static void region_grow( int x, int y, image_double angles, struct point2i * reg,
                         int * reg_size, double * reg_angle, image_char used,
                         double prec );

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).
 */
static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps );

/*----------------------------------------------------------------------------*/
/** Reduce the region size, by elimination the point2is far from the
    starting point2i, until that leads to rectangle with the right
    density of region point2is or to discard the region if too small.
 */
static int reduce_region_radius( struct point2i * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th );

/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

    For that, an estimation of the angle tolerance is performed by the
    standard deviation of the angle at point2is near the region's
    starting point2i. Then, a new region is grown starting from the same
    point2i, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region point2is,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
static int refine( struct point2i * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th );

//--------------------------------------------------------
//my code
bool isArcSegment(point2i * reg, int reg_size, struct rect * main_rect, image_double ll_angles,image_char used,image_char pol,
                         double prec, double p, rect * rect_up, rect * rect_down);


/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD full interface.
 */
double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y );


/*------------------------------------------------------------------------------------------------*/
/**
my code,Alan Lu
����
img  : ����ͼ���һάdouble������,��СΪY*X�����������ȴ洢������ǰ��Ҫӵ���ڴ�
X    : ����ͼ���columns
Y    ������ͼ���rows
���
n_out: lsd�㷨���õ����߶ε�����n��return�ķ���ֵ��n���߶Σ�Ϊһάdouble�����飬����Ϊ8*n��ÿ8��Ϊһ�飬����x1,y1,x2,y2,dx,dy,width,polarity
reg_img: ������������һά��int�����飬��Сreg_y*reg_x,����Ӧ������λ�ñ���������ڵ��߶�(1,2,3,...n),���ֵΪ0��ʾ�������κ��߶�.
         �����ⲿ��int * region_img,��ֻ��Ҫ &region_img,�Ϳ��Եõ��������ķ��أ�����Ҫʱֱ��NULL����
reg_x  : �����������columns,����Ҫʱֱ��NULL����
reg_y  : �����������rows,����Ҫʱֱ��NULL����
*/
double * mylsd(int * n_out, double * img, int X, int Y, int ** reg_img, int * reg_x, int * reg_y);

//lines: �����lines_num���߶Σ�ÿ���߶�8��ֵ������x1,y1,x2,y2,dx,dy,width,polarity
//lines_num:
//new_lines_num: �ܾ����߶κ��new_lines_num���߶Σ�����lines��ǰ�棬���̵��߶λ�ŵ�β�ʹ�
//�˴��������Ʋ�������Ҫ��Ŀǰȡ8^2, 14^2
void     rejectShortLines(double * lines, int lines_num, int * new_lines_num );

/*----------------------------------------------------------------------------*/
//���룺
//start_angle,end_angle, �Ƕȷ�λ��(-pi,pi).  
//  pi    ------->x  0
//        |
//        |
//       y\/ pi/2
//polarity: ��polarityΪ1ʱ����ʾ���Ǵ�start_angle������ʱ�뷽����ת��end_angle�ĽǶ�;��polarityΪ-1ʱ����ʾ���Ǵ�start_angle����˳ʱ�뷽����ת��end_angle�ĽǶ�;
//����ֵ�� ��ת�Ƕ�coverage
inline double rotateAngle(double start_angle, double end_angle, int polarity);

//���߶ΰ���͹�Ժ;�����з���
//lines: �����lines_num���߶Σ�ÿ���߶�8��ֵ������x1,y1,x2,y2,dx,dy,length,polarity
//lines_num:
//�������groups. ÿ������һ��vector<int>
//ע�⣺�м�����region,��Ҫ�ں��������ֶ��ͷ�region
void groupLSs(double *lines, int line_num, int * region, int imgx, int imgy, vector<vector<int>> * groups);

//����groups��ÿ����Ŀ��
//���룺
//lines: �����lines_num���߶Σ�ÿ���߶�8��ֵ������x1,y1,x2,y2,dx,dy,length,polarity
//lines_num:
//groups: ���飬ÿ�����鶼�����߶ε�����
//���:
//coverages: ÿ����Ŀ�ȣ��������߶�ֻ��1��ʱ�����Ϊ0. coverages�ĳ��ȵ���������� = groups.size()
//ע�⣬coverages��ǰ����Ҫ�����ڴ棬coverages�������Ҫ�ں������ֶ��ͷ��ڴ棬���ȵ��ڷ�������
void calcuGroupCoverage(double * lines, int line_num, vector<vector<int>> groups, double * &coverages);

//==============================================================================
//====================================================================================================
//================================clustering==========================================================
//����
//��points�е�i����initializations�е�j����ÿ��Ԫ�ص�ƽ�����ܺ�,ÿ��ά�ȶ�ΪnDims
inline double squaredDifference(int & nDims, double *& points, int & i, double *& initializations, int & j);

/**
 *����
 *points: ����ֵƯ�Ƶĵ㼯���ܹ���nPoints���㣬ÿ������nDimsά�ȣ���һά����
 *initPoints: ��ֵƯ�Ƴ�ʼ��λ�ã���nxd�ռ����Ҿ�ֵƯ�Ƴ�ʼʱ��ʼ������λ�ã��ܹ���initLength���㣬ÿ������nDimsά��
 *sigma = 1
 *window_size: window parameter = distance_tolerance����window parameter = distance_tolerance/2
 *accuracy_tolerance: �����������1e-6
 *iter_times: ��������50
 *���
 *������λ�ã�λ�ø������ʼ������λ�ø���һ��,���ǽ�������µ�initPoints,Ҳ�������������������Ҳ�������������ʡ�ڴ�
 */
void meanShift( double * points, int nPoints, int nDims, double * & initPoints, int initLength, double sigma, double window_size, double accuracy_tolerance, int iter_times );

/***
 *����
 *points,������ĵ㼯,Ϊһά����,nPoints���㣬ÿ����ά����nDims
 *distance_threshold ��������ľ�����ֵ
 *��� outPoints
 *�����ĵ㼯 nOutPoints x nDims 
 *�ú���Ҫǧ��ע�⣬�������ú󣬺����ڲ��������nOutPoints��double�͵������ڴ棬�����ʹ����Ϻ��м�free(outPoints).
 */
void clusterByDistance(double * points, int nPoints, int nDims, double distance_threshold,int number_control, double * & outPoints, int * nOutPoints);


//�����㷨����ֵƯ��
//�������裬һ��ѡȡ��ʼ�����㣬���Ǿ�ֵƯ�ƣ�����ȥ���ظ��㣬�Ӷ��õ���������
//��ú�ѡԲ�ĵľ�������(xi,yi)
//���룺
//points��һά������,����Ϊpoints_num x 2
//distance_tolerance,���ݵ����İ뾶
//�����
//��ά���ݵ�ľ������� centers��һάdouble���飬 ��СΪ centers_num x 2
//��ȷ����ֵΪ1�����ִ���Ϊ0. ����pointsΪ��
//�м��мǣ����� centersΪ�����ڲ�������ڴ棬��������centers_num����ľ������ģ�ʹ�����һ��Ҫ�ͷţ���סfree(centers)������
int  cluster2DPoints( double * points, int points_num, double distance_tolerance, double * & centers, int * centers_num);

//�����㷨����ֵƯ��
//�������裬һ��ѡȡ��ʼ�����㣬���Ǿ�ֵƯ�ƣ�����ȥ���ظ��㣬�Ӷ��õ���������
//��ú�ѡԲ�ĵľ�������(xi,yi)
//���룺
//datas��һά������,����Ϊdatas_num x 1
//distance_tolerance,���ݵ����İ뾶
//�����
//һά���ݵ�ľ������� centers��һάdouble���飬 ��СΪ centers_num x 1
//��ȷ����ֵΪ1�����ִ���Ϊ0. ����pointsΪ��
//�м��мǣ����� centersΪ�����ڲ�������ڴ棬��������centers_num����ľ������ģ�ʹ�����һ��Ҫ�ͷţ���סfree(centers)������
int  cluster1DDatas( double * datas, int datas_num, double distance_tolerance, double * & centers, int * centers_num);

//================================Generate Ellipse Candidates=========================================
//ƥ����ԣ���Ե�������������Բ����
typedef struct PairGroup_s
{
	point2i pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
}PairGroup;

//ƥ����Խڵ�
typedef struct PairGroupNode_s
{
	point2i pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
	PairGroupNode_s* next;
}PairGroupNode;

typedef struct  PairGroupList_s
{
	int length;
	PairGroup * pairGroup;
}PairGroupList;

typedef struct Point2dNode_s
{
	point2d point;
	Point2dNode_s * next;
}Point2dNode;

typedef struct Point3dNode_s
{
	point3d point;
	Point3dNode_s * next;
}Point3dNode;

typedef struct Point5dNode_s
{
	point2d center;
	point2d axis;
	double  phi;
	Point5dNode_s * next;
}Point5dNode;

typedef struct Point1dNode_s
{
	double data;
	Point1dNode_s * next;
}Point1dNode;

PairGroupList * pairGroupListInit( int length);

void freePairGroupList( PairGroupList * list);

//�����ݶȣ�����ģ�ͽǶȣ�ͬʱģֵ̫С�����ص�ֱ�����Ƶ�����ֵΪNOTDEF
//mod��anglesΪ�˴�ֵ���Ƕ���ָ��
void calculateGradient( double * img_in, unsigned int imgx, unsigned int imgy,image_double * mod, image_double * angles);

void calculateGradient2( double * img_in, unsigned int imgx, unsigned int imgy, image_double * angles);

//=============================================================================
//��Ҫ��������ͷ�ļ�
//#include <opencv2\opencv.hpp>
//using namespace cv;
void cvCanny3(	const cv::Mat* srcarr, cv::Mat* dstarr,
				cv::Mat* dxarr, cv::Mat* dyarr,
                int aperture_size );

void Canny3(	InputArray image, OutputArray _edges,
				OutputArray _sobel_x, OutputArray _sobel_y,
                int apertureSize, bool L2gradient );

//canny
void calculateGradient3( double * img_in, unsigned int imgx, unsigned int imgy, image_double * angles);


//=============================================================================
/** Convert ellipse from matrix form to common form:
    ellipse = (centrex,centrey,ax,ay,orientation).
 */
int ellipse2Param(double *p,double param[]);

//input : (xi,yi)
//output: x0,y0,a,b,phi,ellipara��Ҫ���������ڴ�
//successfull, return 1; else return 0
int fitEllipse(point2d* dataxy, int datanum, double* ellipara);

//input: dataxyΪ���ݵ�(xi,yi),�ܹ���datanum��
//output: ��Ͼ���S. ע�⣺S��Ҫ���������ڴ棬double S[36].
inline void calcuFitMatrix(point2d* dataxy, int datanum, double * S);

//input: fit matrixes S1,S2. length is 36.
//output: fit matrix S_out. S_out = S1 + S2.
//S_out������Ҫ�����ڴ�
inline void addFitMatrix(double * S1, double * S2, double * S_out);

//input : S����6 x 6 = 36
//output: (A,B,C,D,E,F)��A>0, ellicoeff��Ҫ���������ڴ�. ��Ҫת����(x0,y0,a,b,phi)ʱ����Ҫ��
//ellipse2Param(ellicoeff,ellipara); ax^2 + bxy + cy^2 + dx + ey + f = 0, transform to (x0,y0,a,b,phi)
//successfull, return 1; else return 0
int fitEllipse2(double * S, double* ellicoeff);

//��Σ�e1 = (x1,y1,a1,b1,phi1), e2 = (x2,y2,a2,b2,phi2)
//��������Ϊ1������Ϊ0
inline bool isEllipseEqual(double * ellipse1, double * ellipse2, double centers_distance_threshold, double semimajor_errorratio, double semiminor_errorratio, double angle_errorratio, double iscircle_ratio);


inline bool regionLimitation( point2d point_g1s, point2d g1s_ls_dir, point2d point_g1e, point2d g1e_ls_dir, point2d point_g2s, point2d g2s_ls_dir, point2d point_g2e, point2d g2e_ls_dir, double polarity, double region_limitation_dis_tolerance);


/*----------------------------------------------------------------------------*/
/** Approximate the distance between a point and an ellipse using Rosin distance.
 */
inline double d_rosin (double *param, double x, double y);

/*----------------------------------------------------------------------------*/

//����
//lsd�㷨���õ����߶μ���lines������line_num��return�ķ���ֵ��line_nums���߶Σ�Ϊһάdouble������lines������Ϊ8*n��ÿ8��Ϊһ��
//����x1,y1,x2,y2,dx,dy,length,polarity
//groups: �߶η��飬ÿ����水�ռ��ηֲ�˳��˳ʱ�������ʱ��洢���߶��������߶�������Χ��0~line_num-1. ����������ָ�룬ʹ��ʱҪע��(*group)
//first_group_ind��second_group_ind��ƥ����ӵ�����������ȡsalient hypothesisʱ��second_group_ind = -1, fit_matrix2 = NULL.
//fit_matrix1, fit_matrix2, �ֱ�����ӵĶ�Ӧ����Ͼ���
//angles, �Ǳ�Ե��ͼ+�ݶȷ��� �ޱ�Ե��ʱ��NODEF
//distance_tolerance:
//group_inliers_num:��¼�Ÿ������֧���ڵ����������飬ʵʱ���£���ʼʱΪ0
//���
//ellipara
bool calcEllipseParametersAndValidate( double * lines, int line_num, vector<vector<int>> * groups, int first_group_ind,int second_group_ind, double * fit_matrix1, double * fit_matrix2, image_double angles, double distance_tolerance, unsigned int * group_inliers_num, point5d *ellipara);


//����
//lsd�㷨���õ����߶μ���lines������line_num��return�ķ���ֵ��line_nums���߶Σ�Ϊһάdouble������lines������Ϊ8*n��ÿ8��Ϊһ��
//����x1,y1,x2,y2,dx,dy,length,polarity
//groups: �߶η��飬ÿ����水�ռ��ηֲ�˳��˳ʱ�������ʱ��洢���߶��������߶�������Χ��0~line_num-1
//coverages: ÿ������ĽǶȸ��Ƿ�Χ0~2pi���������ֻ��1���߶Σ����ǽǶ�Ϊ0�����鳤�ȵ��ڷ����������
//angles ���Ե����ݶȷ���gradient direction, �ޱ�Ե��λNOTDEF
//����ֵ PairedGroupList* list ���ص��ǳ�ʼ��Բ���ϵ����飬����list->length. 
//�мǣ����ڴ��ں��������룬����ú����ǵ��ͷ��ڴ棬���ú���freePairedSegmentList()�����ͷ�

PairGroupList * getValidInitialEllipseSet( double * lines, int line_num, vector<vector<int>> * groups, double * coverages, image_double angles, double distance_tolerance, int specified_polarity);


void generateEllipseCandidates( PairGroupList * pairGroupList, double distance_tolerance, double * & ellipse_candidates, int * candidates_num);
