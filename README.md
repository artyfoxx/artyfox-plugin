# artyfox-plugin
A disjointed set of filters for VapourSynth, I write everything that seems interesting.  
The library is written using AVX2 and FMA3 intrinsics, so processors older than Haswell and Zen 1 are not supported.

## Filters:
* [**BitDepth**](#bitdepth)  
* [**Linearize**](#linearize)  
* [**Transfer**](#transfer)  
* [**Resize**](#resize)  
* [**Descale**](#descale)  
* [**Mean**](#mean)  
* [**Metric**](#metric)  
* [**FixBorder**](#fixborder)  
* [**AverageFields**](#averagefields)
* [**UnsharpMask**](#unsharpmask)

## BitDepth
`artyfox.BitDepth(clip clip, int bits[, bool range="_Range" frame property])`

Converting the bit depth of a clip.
* `clip`: Source clip to be converted to bit depth. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `bits`: The bit depth of the target clip. It can be from `8` to `16` or `32`. When converting from integer to float or vice versa, a color range conversion may also occur, since in the 32-bit float format, the concept of a limited range does not exist. Conversion between integers occurs without regard to range. Downconversion of bit depth occurs with arithmetic rounding and saturation.
* `range`: Specifies the range to use. `True` - full range, `False` - limited range. By default, it is taken from the `"_Range"` frame property.

## Linearize
`artyfox.Linearize(clip clip[, str gamma="_Transfer" frame property])`

Inverse transfer function (linearization) of the specified color space. For YUV, a simplified procedure is used, based on the assumption that the chroma is much less sensitive to gamma correction than the luma.
It is used for subsequent mathematically correct operations with video in linear color space, such as convolution or resizing.
* `clip`: Source clip to linearize. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: The transfer function that was applied to the video and that needs to be inverted. By default, it is taken from the `"_Transfer"` frame property. Possible values:
  * `srgb`: `sRGB` inverse transfer function.
  * `smpte170m`: `SMPTE 170M`, `BT.709`, `BT.601` and `BT.2020` inverse transfer functions.
  * `adobe`: `Adobe RGB` and `opRGB` inverse transfer functions.
  * `dcip3`: `DCI-P3` inverse transfer function.
  * `smpte240m`: `SMPTE 240M` inverse transfer function.
  * `smpte2084`: `SMPTE 2084` (`HDR PQ`) inverse transfer function.

## Transfer
`artyfox.Transfer(clip clip[, str gamma="_Transfer" frame property])`

Transfer function (gamma correction) to the specified color space. For YUV, a simplified procedure is used, based on the assumption that the chroma is much less sensitive to gamma correction than the luma.
It is used for subsequent mathematically correct operations with video in linear color space, such as convolution or resizing.
* `clip`: Source clip for gamma correction. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: The transfer function to be applied to the video. By default, it is taken from the `"_Transfer"` frame property. Possible values:
  * `srgb`: `sRGB` transfer function.
  * `smpte170m`: `SMPTE 170M`, `BT.709`, `BT.601` and `BT.2020` transfer functions.
  * `adobe`: `Adobe RGB` and `opRGB` transfer functions.
  * `dcip3`: `DCI-P3` transfer function.
  * `smpte240m`: `SMPTE 240M` transfer function.
  * `smpte2084`: `SMPTE 2084` (`HDR PQ`) transfer function.

## Resize
`artyfox.Resize(clip clip, int width, int height[, float src_left=0.0, float src_top=0.0, float src_width=clip.width, float src_height=clip.height, str kernel='bilinear', float b=1/3, float c=1/3, float taps=3.0, str confine='inf', str gamma="_Transfer" frame property, float sharp=1.0])`

Implementation of multiple resize functions using double-precision convolution method in a linear color space. For YUV, a simplified procedure is used, based on the assumption that the chroma is much less sensitive to gamma correction than the luma.
* `clip`: Source clip to resize. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `width`: Target width. Must be integer and match the source clip's subsampling.
* `height`: Target height. Must be integer and match the source clip's subsampling.
* `src_left`: The `x` coordinate of the point where the region to be resized starts. Defaults to 0.0
* `src_top`: The `y` coordinate of the point where the region to be resized starts. Defaults to 0.0
* `src_width`: The width of the region to be resized relative to `src_left`. Defaults to the width of the source clip.
* `src_height`: The height of the region to be resized relative to `src_top`. Defaults to the height of the source clip.
* `kernel`: Selecting a kernel for convolution. Possible values:
  * `area`: Area Resize.
  * `bicubic`: Bicubic interpolation.
  * `bilinear`: Bilinear interpolation, used by default.
  * `blackman`: Blackman windowed sinc.
  * `box`: Box interpolation.
  * `gauss`: Gaussian kernel. `p` is specified via the `b` parameter, it must be between 1 and 100, the default value is 30.0. `taps` must be in the range from 1 to 128, the default value is 4.0.
  * `kaiser`: Kaiser–Bessel windowed sinc. `beta` (`Pi` * `alpha`) is specified via the `b` parameter, it must be between 0 and 32, the default value is 4.0.
  * `lanczos`: Lanczos windowed sinc.
  * `magic`: Magic Kernel.
  * `magic13`: Magic Kernel Sharp 2013.
  * `magic21`: Magic Kernel Sharp 2021.
  * `nuttall`: Nuttall windowed sinc.
  * `point`: Nearest neighbour interpolation.
  * `spline16`: Cubic spline with 4 sample points.
  * `spline36`: Cubic spline with 6 sample points.
  * `spline64`: Cubic spline with 8 sample points.
  * `spline100`: Cubic spline with 10 sample points.
  * `spline144`: Cubic spline with 12 sample points.
* `b`: The `b` parameter in the `bicubic` kernel. Defaults to 1/3.
* `c`: The `c` parameter in the `bicubic` kernel. Defaults to 1/3.
* `taps`: Window radius value for `blackman`, `box`, `gauss`, `kaiser`, `lanczos` and `nuttall` kernels, it must be between 1 and 128, the default value is 3.0 (except for `gauss`).
* `confine`: A method for representing pixels that are outside the frame. Possible values:
  * `zero`: Pixels outside the frame are considered zero.
  * `inf`: Pixels outside the frame are replaced with the nearest pixels within the frame, used by default.
  * `mirror`: Pixels outside the frame are considered mirror images of pixels inside the frame.
* `gamma`: The inverse and forward transfer functions. Correction is performed before and after resizing, in order to produce the resize itself in a linear color space. By default, it is taken from the `"_Transfer"` frame property. Possible values:
  * `srgb`: `sRGB` inverse and forward transfer functions.
  * `smpte170m`: `SMPTE 170M`, `BT.709`, `BT.601` and `BT.2020` inverse and forward transfer functions.
  * `adobe`: `Adobe RGB` and `opRGB` inverse and forward transfer functions.
  * `dcip3`: `DCI-P3` inverse and forward transfer functions.
  * `smpte240m`: `SMPTE 240M` inverse and forward transfer functions.
  * `smpte2084`: `SMPTE 2084` (`HDR PQ`) inverse and forward transfer functions.
  * `none`: completely disables transfer, resizing occurs directly, in a logarithmic color space.
* `sharp`: Optional post sharp. It is performed after resizing, but before gamma correction. By default, 1.0 (sharp is disabled). Values ​​​​less than 1.0 - blur, more - sharp. The allowed range of values ​​is from 0.1 to 5.0

Chroma alignment in YUV with subsampling is performed based on the `"_ChromaLocation"` property. Resize and alignment by fields are not supported.

## Descale
`artyfox.Descale(clip clip, int width, int height[, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, str kernel='bilinear', float b=1/3, float c=1/3, float taps=3.0, str confine='inf', float reg=1e-8])`

Descaling via Tikhonov regularization and Cholesky decomposition (U.T @ U) using double-precision convolution method.
* `clip`: Source clip to descale. Must be RGB, YUV or GRAY. 32-bit float sample type only.
* `width`: Target width. Must be integer and match the source clip's subsampling.
* `height`: Target height. Must be integer and match the source clip's subsampling.
* `src_left`: The `x` coordinate of the point where the destination region after descaling starts. Defaults to 0.0
* `src_top`: The `y` coordinate of the point where the destination region after descaling starts. Defaults to 0.0
* `src_width`: The width of the destination region after descaling relative to `src_left`. Defaults to `width`.
* `src_height`: The height of the destination region after descaling relative to `src_top`. Defaults to `height`.
* `kernel`: Selecting a kernel for deconvolution. Possible values:
  * `area`: Area Resize.
  * `bicubic`: Bicubic interpolation.
  * `bilinear`: Bilinear interpolation, used by default.
  * `blackman`: Blackman windowed sinc.
  * `box`: Box interpolation.
  * `gauss`: Gaussian kernel. `p` is specified via the `b` parameter, it must be between 1 and 100, the default value is 30.0. `taps` must be in the range from 1 to 128, the default value is 4.0.
  * `kaiser`: Kaiser–Bessel windowed sinc. `beta` (`Pi` * `alpha`) is specified via the `b` parameter, it must be between 0 and 32, the default value is 4.0.
  * `lanczos`: Lanczos windowed sinc.
  * `magic`: Magic Kernel.
  * `magic13`: Magic Kernel Sharp 2013.
  * `magic21`: Magic Kernel Sharp 2021.
  * `nuttall`: Nuttall windowed sinc.
  * `point`: Nearest neighbour interpolation.
  * `spline16`: Cubic spline with 4 sample points.
  * `spline36`: Cubic spline with 6 sample points.
  * `spline64`: Cubic spline with 8 sample points.
  * `spline100`: Cubic spline with 10 sample points.
  * `spline144`: Cubic spline with 12 sample points.
* `b`: The `b` parameter in the `bicubic` kernel. Defaults to 1/3.
* `c`: The `c` parameter in the `bicubic` kernel. Defaults to 1/3.
* `taps`: Window radius value for `blackman`, `box`, `gauss`, `kaiser`, `lanczos` and `nuttall` kernels, it must be between 1 and 128, the default value is 3.0 (except for `gauss`).
* `confine`: A method for representing pixels that are outside the frame. Possible values:
  * `zero`: Pixels outside the frame are considered zero.
  * `inf`: Pixels outside the frame are replaced with the nearest pixels within the frame, used by default.
  * `mirror`: Pixels outside the frame are considered mirror images of pixels inside the frame.
* `reg`: Regularization parameter. Ensures positive definiteness and stability of the solution to the system of equations. Small values ​​can lead to increased noise, quantization artifacts, and ringing. Excessively large values ​​produce a smooth image, suppressing fine details. Default is 1e-8. Valid range: 1e-16 <= `reg` < 1.

## Mean
`artyfox.Mean(clip clip[, str mode='am', int plane=0, bool norm=True])`

Calculates the `minimum`, `maximum` and the set mean value for the specified plane of each frame of the clip and stores them in the frame properties. For the 32-bit float sample type, negative pixel values ​​are handled correctly in the `am`, `median`, `limsad`, `simsad`, `tbmsad` and `adtbm` modes; the other modes are not suitable for this usage case due to the mathematics used.
* `clip`: Original clip. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `mode`: Algorithm for finding the mean value of a frame plane. Can take and return the following values:
  * `am`: Arithmetic mean (`arithmetic_mean`). Used by default.
  * `gm`: Geometric mean (`geometric_mean`). Vector log is implemented without any checks; if x<=0, it returns incorrect results, so it is necessary to manually limit the input range.
  * `agm`: Arithmetic-geometric mean (`arithmetic_geometric_mean`).
  * `hm`: Harmonic mean (`harmonic_mean`). Since division by zero is inf, the lower range for integer is limited to `1`, and for float to `1e-16`.
  * `chm`: Contraharmonic mean (`contraharmonic_mean`).
  * `rms`: Root mean square (`root_mean_square`).
  * `rmc`: Root mean cube (`root_mean_cube`).
  * `median`: Median (`median`).
  * `limsad`: Modified sum of absolute differences between the mutual linear (`2 points`) interpolation of the frame fields and the true values ​​of the corresponding pixels; the `minimum` and `maximum` correspond to the minimum and maximum absolute difference between the interpolated and true values ​​of the frame (`linear_interp_msad`).
  * `simsad`: Modified sum of absolute differences between the mutual stable (`6 points`) interpolation of the frame fields and the true values ​​of the corresponding pixels; the `minimum` and `maximum` correspond to the minimum and maximum absolute difference between the interpolated and true values ​​of the frame (`stable_interp_msad`).
  * `tbmsad`: Modified sum of absolute differences between the even and odd lines of the frame; the `minimum` and `maximum` values ​​correspond to the minimum and maximum absolute difference between even and odd lines of the frame (`top_bottom_msad`).
  * `adtbm`: The absolute difference between the arithmetic means of the top and bottom fields of the frame; the `minimum` and `maximum` are the absolute difference between the minimums and maximums of both fields (`abs_diff_top_bottom_means`).
* `plane`: Plane for analysis. Default is 0.
* `norm`: Normalizes the mean to the 0:1 range (only for integer sample types). Default is true.

## Metric
`artyfox.Metric(clip clip0, clip clip1[, str mode='relative', float thr=0.0])`

Compares two video clips and calculates the specified difference metric. The metric is stored as a frame property. The input range is always clamped to 0:1.
* `clip0`: Original clip. Must be GRAY. 32-bit float sample type only.
* `clip1`: Restored clip. Must be GRAY. 32-bit float sample type only. The width, height and number of frames must match the original clip.
* `mode`: Algorithm for finding the difference between two clips. Can take and return the following values:
  * `relative`: Relative error (`RelativeError`). Used by default.
  * `rmse`: Root mean square error (`RMSE`).
  * `psnr`: Peak signal-to-noise ratio (`PSNR`).
  * `msad`: Modified sum of absolute differences (`MSAD`).
  * `pcc`: Pearson correlation coefficient (`PCC`).
  * `ssim`: Structural similarity index measure (`SSIM`).
* `thr`: The absolute difference threshold above which the difference is considered significant; everything else is set to zero. This allows noise to be filtered out. Doesn't work for `pcc` and `ssim`. The default is 0.

## FixBorder
`artyfox.FixBorder(clip clip, str[] fix)`

Function for correcting brightness artifacts at frame borders.
* `clip`: The clip that needs correction.
* `fix`: A list containing strings with instructions for correction. The string has the following format: "`plane` `axis` `target` `donor` `limit` `shift` `clamp`". The first four values are mandatory.
  * `plane`: The target plane.
  * `axis`: The axis along which the correction is performed. 'x' - columns, 'y' - rows.
  * `target`: The target column/row that needs correction. Counted from the upper left corner of the frame. Can be an integer or several comma-separated numbers (no more than 256). Can be negative, in which case it is counted from the lower right corner.
  * `donor`: The donor column/row on which the correction is based. Counted from the upper left corner of the frame. Can be an integer or several comma-separated numbers (no more than 256). Can be negative, in which case it is counted from the lower right corner.
  * `limit`: Limit on the maximum brightness change. If the value is positive, the brightness cannot rise above or fall, if negative, it cannot fall below the specified value or rise. Specified in 8-bit notation. Default is 0 (no limit). The allowed range of values ​​is from -255 to 255.
  * `shift`: Shift the zero point of the correction curve relative to the beginning of the range. For YUV, it applies to luma only. Specified in 8-bit notation. Default is 0.0. The allowed range of values ​​is from -19.0 to 279.0. Note: The function that converts a string to a double can take the fraction separator from the system locale settings. If a period separator causes an error, use the separator set in your system.
  * `clamp`: Clamps the brightness of the corrected target column/row between the maximum and minimum of the donor column/row. Defaults to True.

## AverageFields
`artyfox.AverageFields(clip clip[, float weight=0.5, float shift=0.0])`

A function for correcting interlaced fades. It works by averaging the fields of a frame based on their average brightness ratio.
* `clip`: The clip that needs correction.
* `weight`: The ratio of the contribution of fields to the resulting clip. 0.0 means the top field remains unchanged, and the bottom field is adjusted based on the top field. 1.0 means the bottom field remains unchanged, and the top field is adjusted based on the bottom field. Anything in between means the fields are mutually adjusted based on the set ratio. Default is 0.5. The allowed range of values ​​is from 0.0 to 1.0.
* `shift`: Shift the zero point of the correction curve relative to the beginning of the range. For YUV, it applies to luma only. Specified in 8-bit notation. Default is 0.0. The allowed range of values ​​is from -19.0 to 279.0.

## UnsharpMask
`artyfox.UnsharpMask(clip clip[, int strength=64, int radius=3, int threshold=8, str mode='box', int passes=1, bool rounding=False, int[] planes=[0, 1, 2] if clip.format.color_family == vs.RGB else 0])`

A port of `UnsharpMask` from `AviSynth` with a few additions. By default, it completely replicates the original filter's algorithm.
* `clip`: Target clip to receive the unsharp mask.
* `strength`: Adjusts the amount of enhancement and attenuation of the unsharp mask effect. Valid range: 1 to 512. Default is 64.
* `radius`: The blur radius used to create the unsharp mask. Valid range: 1 to 127. Default is 3.
* `threshold`: A threshold for the absolute difference between the original and blurred clips, above which the difference is considered significant. Anything less than or equal to this value is passed unchanged. This allows for noise filtering. Specified in 8-bit notation. Valid range: 0 to 255. Default is 8.
* `mode`: Blur mode:
  * `box`: Box blur, used by default.
  * `stack`: Stack blur (triangular core).
* `passes`: The number of blur passes. This allows to emulate a Gaussian kernel with the desired degree of approximation. `passes=2, mode='box'` corresponds to `passes=1, mode='stack'`, taking into account the difference in rounding. Valid range: 1 to 16. Default is 1.
* `rounding`: Rounding mode. `False` - round down (floor), `True` - arithmetic rounding. Default is `False`.
* `planes`: List of planes for the unsharp mask. Defaults to `all` for RGB and `0` for other color families.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
