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
* [**MedianBlur**](#medianblur)
* [**RemoveGrain**](#removegrain)
* [**Repair**](#repair)
* [**Convolution**](#convolution)
* [**Expand**](#expand)
* [**Inpand**](#inpand)
* [**Inflate**](#inflate)
* [**Deflate**](#deflate)
* [**Clamp**](#clamp)
* [**MakeDiff**](#makediff)
* [**MergeDiff**](#mergediff)
* [**Binarize**](#binarize)

## BitDepth
`artyfox.BitDepth(clip clip, int bits[, bool range="_Range" frame property])`

Converting the bit depth of a clip.
* `clip`: Source clip to be converted to bit depth. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `bits`: The bit depth of the target clip. It can be from `8` to `16` or `32`. When converting from integer to float or vice versa, a color range conversion may also occur, since in the 32-bit float format, the concept of a limited range does not exist. Conversion between integers occurs without regard to range. Downconversion of bit depth from float to integer is done by banker's rounding (half to even), from integer to integer by arithmetic rounding (half up).
* `range`: Specifies the range to use. `True` - full range, `False` - limited range. By default, it is taken from the `"_Range"` frame property.

## Linearize
`artyfox.Linearize(clip clip[, str gamma="_Transfer" frame property])`

Inverse transfer function (linearization) of the specified color space. For YUV, a simplified procedure is used, based on the assumption that the chroma is much less sensitive to gamma correction than the luma.
It is used for subsequent mathematically correct operations with video in linear color space, such as convolution or resizing.
* `clip`: Source clip to linearize. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: The transfer function that was applied to the video and that needs to be inverted. By default, it is taken from the `"_Transfer"` frame property. Possible values:
  * `srgb`: `sRGB` inverse transfer function.
  * `smpte170m`: `SMPTE 170M`, `BT.601`, `BT.709` and `BT.2020` inverse transfer functions.
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
  * `smpte170m`: `SMPTE 170M`, `BT.601`, `BT.709` and `BT.2020` transfer functions.
  * `adobe`: `Adobe RGB` and `opRGB` transfer functions.
  * `dcip3`: `DCI-P3` transfer function.
  * `smpte240m`: `SMPTE 240M` transfer function.
  * `smpte2084`: `SMPTE 2084` (`HDR PQ`) transfer function.

## Resize
`artyfox.Resize(clip clip, int width, int height[, float src_left=0.0, float src_top=0.0, float src_width=clip.width, float src_height=clip.height, str kernel='bilinear', float b=1/3, float c=1/3, float taps=3.0, str confine='inf', str gamma="_Transfer" frame property])`

Implementation of multiple resize functions using double-precision convolution method in a linear color space. For YUV, a simplified procedure is used, based on the assumption that the chroma is much less sensitive to gamma correction than the luma.
* `clip`: Source clip to resize. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `width`: Target width. Must be integer and match the source clip's subsampling.
* `height`: Target height. Must be integer and match the source clip's subsampling.
* `src_left`: The `x` coordinate of the point where the region to be resized starts. Defaults to `0.0`
* `src_top`: The `y` coordinate of the point where the region to be resized starts. Defaults to `0.0`
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
* `b`: The `b` parameter in the `bicubic` kernel. Defaults to `1/3`.
* `c`: The `c` parameter in the `bicubic` kernel. Defaults to `1/3`.
* `taps`: Window radius value for `blackman`, `box`, `gauss`, `kaiser`, `lanczos` and `nuttall` kernels, it must be between 1 and 128, the default value is `3.0` (except for `gauss`).
* `confine`: A method for representing pixels that are outside the frame. Possible values:
  * `zero`: Pixels outside the frame are considered zero.
  * `inf`: Pixels outside the frame are replaced with the nearest pixels within the frame, used by default.
  * `mirror`: Pixels outside the frame are considered mirror images of pixels inside the frame.
* `gamma`: The inverse and forward transfer functions. Correction is performed before and after resizing, in order to produce the resize itself in a linear color space. By default, it is taken from the `"_Transfer"` frame property. Possible values:
  * `srgb`: `sRGB` inverse and forward transfer functions.
  * `smpte170m`: `SMPTE 170M`, `BT.601`, `BT.709` and `BT.2020` inverse and forward transfer functions.
  * `adobe`: `Adobe RGB` and `opRGB` inverse and forward transfer functions.
  * `dcip3`: `DCI-P3` inverse and forward transfer functions.
  * `smpte240m`: `SMPTE 240M` inverse and forward transfer functions.
  * `smpte2084`: `SMPTE 2084` (`HDR PQ`) inverse and forward transfer functions.
  * `none`: completely disables transfer, resizing occurs directly, in a logarithmic color space.

Chroma alignment in YUV with subsampling is performed based on the `"_ChromaLocation"` property. Resize and alignment by fields are not supported.

## Descale
`artyfox.Descale(clip clip, int width, int height[, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, str kernel='bilinear', float b=1/3, float c=1/3, float taps=3.0, str confine='inf', float reg=1e-8])`

Descaling via Tikhonov regularization and Cholesky decomposition (U.T @ U) using double-precision convolution method.
* `clip`: Source clip to descale. Must be RGB, YUV or GRAY. 32-bit float sample type only.
* `width`: Target width. Must be integer and match the source clip's subsampling.
* `height`: Target height. Must be integer and match the source clip's subsampling.
* `src_left`: The `x` coordinate of the point where the destination region after descaling starts. Defaults to `0.0`
* `src_top`: The `y` coordinate of the point where the destination region after descaling starts. Defaults to `0.0`
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
* `b`: The `b` parameter in the `bicubic` kernel. Defaults to `1/3`.
* `c`: The `c` parameter in the `bicubic` kernel. Defaults to `1/3`.
* `taps`: Window radius value for `blackman`, `box`, `gauss`, `kaiser`, `lanczos` and `nuttall` kernels, it must be between 1 and 128, the default value is `3.0` (except for `gauss`).
* `confine`: A method for representing pixels that are outside the frame. Possible values:
  * `zero`: Pixels outside the frame are considered zero.
  * `inf`: Pixels outside the frame are replaced with the nearest pixels within the frame, used by default.
  * `mirror`: Pixels outside the frame are considered mirror images of pixels inside the frame.
* `reg`: Regularization parameter. Ensures positive definiteness and stability of the solution to the system of equations. Small values ​​can lead to increased noise, quantization artifacts, and ringing. Excessively large values ​​produce a smooth image, suppressing fine details. Default is `1e-8`. Valid range: 1e-16 <= `reg` < 1.

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
* `plane`: Plane for analysis. Default is `0`.
* `norm`: Normalizes the mean to the 0:1 range (only for integer sample types). Default is `True`.

## Metric
`artyfox.Metric(clip clip0, clip clip1[, str mode='relative', float thr=0.0])`

Compares two video clips and calculates the specified difference metric. The metric is stored as a frame property. The input range is always clamped to 0.0...1.0.
* `clip0`: Original clip. Must be GRAY. 32-bit float sample type only.
* `clip1`: Restored clip. Must be GRAY. 32-bit float sample type only. The width, height and number of frames must match the original clip.
* `mode`: Algorithm for finding the difference between two clips. Can take and return the following values:
  * `relative`: Relative error (`RelativeError`). Used by default.
  * `rmse`: Root mean square error (`RMSE`).
  * `psnr`: Peak signal-to-noise ratio (`PSNR`).
  * `msad`: Modified sum of absolute differences (`MSAD`).
  * `pcc`: Pearson correlation coefficient (`PCC`).
  * `ssim`: Structural similarity index measure (`SSIM`).
* `thr`: The absolute difference threshold above which the difference is considered significant; everything else is set to zero. This allows noise to be filtered out. Doesn't work for `pcc` and `ssim`. The default is `0`.

## FixBorder
`artyfox.FixBorder(clip clip, str[] fix)`

Function for correcting brightness artifacts at frame borders.
* `clip`: The clip that needs correction. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `fix`: A list containing strings with instructions for correction. The string has the following format: "`plane` `axis` `target` `donor` `limit` `shift` `clamp`". The first four values are mandatory.
  * `plane`: The target plane.
  * `axis`: The axis along which the correction is performed. 'x' - columns, 'y' - rows.
  * `target`: The target column/row that needs correction. Counted from the upper left corner of the frame. Can be an integer or several comma-separated numbers. Can be negative, in which case it is counted from the lower right corner.
  * `donor`: The donor column/row on which the correction is based. Counted from the upper left corner of the frame. Can be an integer or several comma-separated numbers. Can be negative, in which case it is counted from the lower right corner.
  * `limit`: Limit on the maximum brightness change. If the value is positive, the brightness cannot rise above or fall, if negative, it cannot fall below the specified value or rise. Specified in 8-bit notation. Default is `0` (no limit). The allowed range of values ​​is from -255 to 255.
  * `shift`: Shift the zero point of the correction curve relative to the beginning of the range. For YUV, it applies to luma only. Specified in 8-bit notation. Default is `0.0`. The allowed range of values ​​is from -19.0 to 279.0.
  * `clamp`: Clamps the brightness of the corrected target column/row between the maximum and minimum of the donor column/row. Default is `1` (true).

## AverageFields
`artyfox.AverageFields(clip clip[, float weight=0.5, float shift=0.0])`

A function for correcting interlaced fades. It works by averaging the fields of a frame based on their average brightness ratio.
* `clip`: The clip that needs correction. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `weight`: The ratio of the contribution of fields to the resulting clip. 0.0 means the top field remains unchanged, and the bottom field is adjusted based on the top field. 1.0 means the bottom field remains unchanged, and the top field is adjusted based on the bottom field. Anything in between means the fields are mutually adjusted based on the set ratio. Default is `0.5`. The allowed range of values ​​is from 0.0 to 1.0.
* `shift`: Shift the zero point of the correction curve relative to the beginning of the range. For YUV, it applies to luma only. Specified in 8-bit notation. Default is `0.0`. The allowed range of values ​​is from -19.0 to 279.0.

## UnsharpMask
`artyfox.UnsharpMask(clip clip[, int strength=64, int radius=3, int threshold=8, str mode='box', int passes=1, bool rounding=False, int[] planes=[0, 1, 2] if clip.format.color_family == vs.RGB else 0])`

A port of `UnsharpMask` from `AviSynth` with a few additions. By default, it completely replicates the original filter's algorithm.
* `clip`: Target clip to receive the unsharp mask. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `strength`: Adjusts the amount of enhancement and attenuation of the unsharp mask effect. Valid range: 1 to 512. Default is `64`.
* `radius`: The blur radius used to create the unsharp mask. Valid range: 1 to 127. Default is `3`.
* `threshold`: A threshold for the absolute difference between the original and blurred clips, above which the difference is considered significant. Anything less than or equal to this value is passed unchanged. This allows for noise filtering. Specified in 8-bit notation. Valid range: 0 to 255. Default is `8`.
* `mode`: Blur mode:
  * `box`: Box blur, used by default.
  * `stack`: Stack blur (triangular core).
* `passes`: The number of blur passes. This allows to emulate a Gaussian kernel with the desired degree of approximation thanks to the central limit theorem. `passes=2, mode='box'` is equivalent to `passes=1, mode='stack'`, taking into account the difference in internal rounding. Valid range: 1 to 16. Default is `1`.
* `rounding`: Rounding mode. `False` - round down (floor), `True` - round half up. Default is `False`.
* `planes`: List of planes for the unsharp mask. Defaults to `[0, 1, 2]` for RGB and `0` for other color families.

## MedianBlur
`artyfox.MedianBlur(clip clip[, int radius=2, int[] planes=[0, 1, 2]])`

Median blur with mathematically correct edge processing.
* `clip`: The clip to which the median blur should be applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `radius`: Blur radius. Radii 1-3 are implemented as separate SIMD functions. The rest are done through universal scalar functions using Huang's algorithm (for 8 bits the histogram is built directly, for 16 bits the Fenwick tree is used, for 32 bits the Treap is used). Valid range: 1 to 127. Default is `2`.
* `planes`: List of planes to be median blurred. By default, all planes are processed.

## RemoveGrain
`artyfox.RemoveGrain(clip clip[, int[] mode=[2, 2, 2]])`

Purely spatial denoising function that includes 30 different modes. Mode 13 through 16 are meant for deinterlacing only, they required interlaced content. In general, the filter is similar to that in the RgTool package, and all changes are aimed at increasing the accuracy and speed of the filter.
* `clip`: The clip to which noise reduction should be applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `mode`: Noise reduction mode. Can be set separately for each plane. If there are more planes than modes specified, the last mode is copied to the remaining planes. Most modes work with 3x3 blocks with the target pixel in the center of the block. Modes 13-16 work with two rows of three pixels, located above and below the target pixel. The edges of planes with a thickness of one pixel are not processed. Supported modes:
  * `0`: The plane is copied without changes.
  * `1`: Clips the pixel with the minimum and maximum of the 8 neighbour pixels.
  * `2`: Clips the pixel with the second minimum and maximum of the 8 neighbour pixels, used by default.
  * `3`: Clips the pixel with the third minimum and maximum of the 8 neighbour pixels. Sames as mode 2 but rounded up to third minimum value (but artifact risky).
  * `4`: Clips the pixel with the fourth minimum and maximum of the 8 neighbour pixels, which is equivalent to a median filter. Sames as mode 2 but rounded up to fourth minimum value (but artifact risky).
  * `5`: Line-sensitive clipping giving the minimal change. Edge sensitive. Only line pairs are used. Strong edge protection.
  * `6`: Line-sensitive clipping, intermediate. Edge sensitive. Only line pairs are used. Fairly edge protection.
  * `7`: Line-sensitive clipping, intermediate. Edge sensitive. Only line pairs are used. Mild edge protection.
  * `8`: Line-sensitive clipping, intermediate. Edge sensitive. Only line pairs are used. Faint edge protection.
  * `9`: Line-sensitive clipping on a line where the neighbours pixels are the closest.
  * `10`: Replaces the target pixel with the closest neighbour. Minimal sharpening.
  * `11`: [1, 2, 1] horizontal and vertical kernel blur.
  * `12`: Same as mode 11, but faster and less accurate (modes 11 and 12 from RgTool).
  * `13`: Bob mode, interpolates top field from the line where the neighbour pixels are the closest.
  * `14`: Bob mode, interpolates bottom field from the line where the neighbour pixels are the closest.
  * `15`: Bob mode, interpolates top field. Same as 13 but with a more complicated interpolation formula.
  * `16`: Bob mode, interpolates bottom field. Same as 14 but with a more complicated interpolation formula.
  * `17`: Clips the pixel with the minimum and maximum of respectively the maximum and minimum of each pair of opposite neighbor pixels. Same as mode 4 but better edge protection (similar to near artifact free mode 2).
  * `18`: Line-sensitive clipping using opposite neighbours whose greatest distance from the current pixel is minimal. Same as mode 9 but better edge protection (same as what mode 17 was to mode 4, but in this case to mode 9, and far less denoising than mode 17).
  * `19`: Blur. Replaces the pixel with the average of its 8 neighbours.
  * `20`: Blur. Averages the 9 pixels ([1, 1, 1] horizontal and vertical blur).
  * `21`: Clips pixels using the averages of opposite neighbour. Clipping is done with respect to averages of neighbours. Best for cartoons.
  * `22`: Same as mode 21 but simpler and faster.
  * `23`: Small edge and halo removal, but reputed useless. Fixes small (as one pixel wide) haloes.
  * `24`: Small edge and halo removal, but reputed useless. Same as 23 but considerably more conservative and slightly slower.
  * `25`: Minimal sharpening.
  * `26`: Clips the pixel with the minimum and maximum of respectively the maximum and minimum of pixel pairs. Based off mode 17, but preserves corners, but not thin lines.
  * `27`: Clips the pixel with the minimum and maximum of respectively the maximum and minimum of pixel pairs. Similar to 26 but with 12 pixel pairs instead of 8.
  * `28`: Clips the pixel with the minimum and maximum of respectively the maximum and minimum of pixel pairs. Similar to 27 but with a bit different pixel pairs. Usually no visual difference from mode 27.
  * `29`: SmartRG18
  * `30`: SoftRG18

For the float sample type, no preliminary preparation of the input data, such as aligning the chroma range with the luma or trimming the range to values ​​acceptable by the standard, is performed.

## Repair
`artyfox.Repair(clip clip, clip repairclip[, int[] mode=[2, 2, 2]])`

Repairs unwanted artifacts from (but not limited to) RemoveGrain.
* `clip`: Input clip; known as the reference clip and it's usually the original unprocessed clip. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `repairclip`: Input clip to be repaired; usually the processed clip. The width, height, format and number of frames must match the original clip.
* `mode`: These modes are similar to the RemoveGrain modes but include the center pixel of the reference clip for min/max calculation. Supported modes:
  * `0`: The plane is copied without changes.
  * `1`: Clips the source pixel with the Nth minimum and maximum found on the 3×3-pixel square from the reference clip.
  * `2`: Clips the source pixel with the Nth minimum and maximum found on the 3×3-pixel square from the reference clip, used by default.
  * `3`: Clips the source pixel with the Nth minimum and maximum found on the 3×3-pixel square from the reference clip.
  * `4`: Clips the source pixel with the Nth minimum and maximum found on the 3×3-pixel square from the reference clip.
  * `5`: Line-sensitive clipping giving the minimal change.
  * `6`: Line-sensitive clipping, intermediate.
  * `7`: Line-sensitive clipping, intermediate.
  * `8`: Line-sensitive clipping, intermediate.
  * `9`: Line-sensitive clipping on a line where the neighbor pixels are the closest.
  * `10`: Replaces the target pixel with the closest pixel from the 3×3-pixel reference square.
  * `11`: Same as modes 1–4 but uses min(Nth_min, c) and max(Nth_max, c) for the clipping, where c is the value of the center pixel of the reference clip.
  * `12`: Same as modes 1–4 but uses min(Nth_min, c) and max(Nth_max, c) for the clipping, where c is the value of the center pixel of the reference clip.
  * `13`: Same as modes 1–4 but uses min(Nth_min, c) and max(Nth_max, c) for the clipping, where c is the value of the center pixel of the reference clip.
  * `14`: Same as modes 1–4 but uses min(Nth_min, c) and max(Nth_max, c) for the clipping, where c is the value of the center pixel of the reference clip.
  * `15`: Clips the source pixels using a clipping pair from the RemoveGrain modes 5 and 6.
  * `16`: Clips the source pixels using a clipping pair from the RemoveGrain modes 5 and 6.
  * `17`: Clips the source pixels using a clipping pair from the RemoveGrain modes 17 and 18.
  * `18`: Clips the source pixels using a clipping pair from the RemoveGrain modes 17 and 18.
  * `19`: ??
  * `20`: ??
  * `21`: ??
  * `22`: ??
  * `23`: ??
  * `24`: ??
  * `25`: Not available. The plane is copied without changes.
  * `26`: Clips the source pixels using a clipping pair from the RemoveGrain mode 26.
  * `27`: Clips the source pixels using a clipping pair from the RemoveGrain mode 27.
  * `28`: Clips the source pixels using a clipping pair from the RemoveGrain mode 28.

For the float sample type, no preliminary preparation of the input data, such as aligning the chroma range with the luma or trimming the range to values ​​acceptable by the standard, is performed.

## Convolution
`artyfox.Convolution(clip clip[, int[] matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], str mode='square', float divisor=sum(matrix), bool saturate=True, int[] planes=[0, 1, 2]])`

Spatial convolution with additional modes. Unlike standard convolution: there is no bias, the weights in the matrix are only integer, infinity borders are like in `mt_convolution`, the values of the weights and the size of the matrix are limited only by common sense.
* `clip`: The clip to which spatial convolution will be applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `matrix`: The weight matrix for convolution. The number of weights must be odd. For a square matrix, the size of the square's side must also be odd. By default, a 3x3 box blur is used.
* `mode`: Convolution operating mode. Can take the following values:
  * `square`: The matrix is ​​interpreted as square. A 3x3 matrix is processed by a specialized SIMD function, while matrices of larger dimensions are processed by a universal scalar function, used by default.
  * `vertical`: The matrix is ​​interpreted as vertical. Processed by a universal SIMD function.
  * `horizontal`: Матрица интерпретируется как горизонтальная. Processed by a universal SIMD function.
  * `both`: The matrix is ​​interpreted first as vertical, then as horizontal.
  * `sobel`: The `sobel` operator from `mt_edge`. Just a preset for `square`. `matrix`, `divisor`, and `saturate` are ignored.
  * `roberts`: The `roberts` operator from `mt_edge`. Just a preset for `square`. `matrix`, `divisor`, and `saturate` are ignored.
  * `laplace`: The `laplace` operator from `mt_edge`. Just a preset for `square`. `matrix`, `divisor`, and `saturate` are ignored.
  * `cartoon`: The `cartoon` operator from `mt_edge`. Just a preset for `square`. `matrix`, `divisor`, and `saturate` are ignored.
  * `min/max`: The `min/max` operator from `mt_edge`. Processed by a specialized SIMD function. `matrix`, `divisor`, and `saturate` are ignored.
  * `hprewitt`: The `hprewitt` operator from `mt_edge`. Processed by a specialized SIMD function. `matrix`, `divisor`, and `saturate` are ignored.
  * `prewitt`: The `prewitt` operator from `mt_edge`. Processed by a specialized SIMD function. `matrix`, `divisor`, and `saturate` are ignored.
  * `kirsch4`: The four-way `kirsch` operator. Processed by a specialized SIMD function. `matrix`, `divisor`, and `saturate` are ignored.
  * `kirsch8`: The eight-way `kirsch` operator. Processed by a specialized SIMD function. `matrix`, `divisor`, and `saturate` are ignored.
* `divisor`: Divisor for the convolution result. All divisions are performed in `float`, rounding to integers is done using banker's method. By default, it is equal to the sum of the matrix weights or `1.0` if the sum is zero.
* `saturate`: Saturates the division result to within the acceptable format limits. `True` - saturation of the result (used by default), `False` - saturation of the absolute value of the result. For the float sample type, saturation is not performed, only the absolute value is taken.
* `planes`: List of planes to be convolved. By default, all planes are processed.

For the float sample type, no preliminary preparation of the input data, such as aligning the chroma range with the luma or trimming the range to values ​​acceptable by the standard, is performed.

## Expand
`artyfox.Expand(clip clip[, int thr=255, str mode='square', int[] planes=[0, 1, 2]])`

Analogue of `mt_expand` from `MaskTools`. Replaces a pixel with a local maximum of pixels in a neighborhood of unit radius.
* `clip`: The clip to which the filter is applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `thr`: Limits the brightness increase of pixels to the specified threshold. Specified in 8-bit notation. Valid range: 1 to 255. Default is `255`.
* `mode`: Specifies the neighbor selection mode for local maximum search. It can take the following values:
  * `square`: 3x3 square neighbourhood, used by default.
  * `horizontal`: 3x1 horizontal neighbourhood.
  * `vertical`: 1x3 vertical neighbourhood.
  * `both`: A 3-long cross (`horizontal` + `vertical`) neighbourhood.
  * `0 -1 -1 0 0 0 1 0 0 1`: Custom mode, in this case analogous to `both`. Pixels are specified as relative coordinates along the `x` and `y` axes. Pixels outside the frame are replaced with the nearest pixels within the frame.
* `planes`: List of planes to be filtered. By default, all planes are processed.

## Inpand
`artyfox.Inpand(clip clip[, int thr=255, str mode='square', int[] planes=[0, 1, 2]])`

Analogue of `mt_inpand` from `MaskTools`. Replaces a pixel with a local minimum of pixels in a neighborhood of unit radius.
* `clip`: The clip to which the filter is applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `thr`: Limits the brightness decrease of pixels to the specified threshold. Specified in 8-bit notation. Valid range: 1 to 255. Default is `255`.
* `mode`: Specifies the neighbor selection mode for local minimum search. It can take the following values:
  * `square`: 3x3 square neighbourhood, used by default.
  * `horizontal`: 3x1 horizontal neighbourhood.
  * `vertical`: 1x3 vertical neighbourhood.
  * `both`: A 3-long cross (`horizontal` + `vertical`) neighbourhood.
  * `0 -1 -1 0 0 0 1 0 0 1`: Custom mode, in this case analogous to `both`. Pixels are specified as relative coordinates along the `x` and `y` axes. Pixels outside the frame are replaced with the nearest pixels within the frame.
* `planes`: List of planes to be filtered. By default, all planes are processed.

## Inflate
`artyfox.Inflate(clip clip[, int thr=255, int[] planes=[0, 1, 2]])`

Analogue of `mt_inflate` from `MaskTools`. Calculates the local average value, taking into account only neighbors whose value is higher than the pixel value.
* `clip`: The clip to which the filter is applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `thr`: Limits the brightness increase of pixels to the specified threshold. Specified in 8-bit notation. Valid range: 1 to 255. Default is `255`.
* `planes`: List of planes to be filtered. By default, all planes are processed.

## Deflate
`artyfox.Deflate(clip clip[, int thr=255, int[] planes=[0, 1, 2]])`

Analogue of `mt_deflate` from `MaskTools`. Calculates the local average value, taking into account only neighbors whose value is lower than the pixel value.
* `clip`: The clip to which the filter is applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `thr`: Limits the brightness decrease of pixels to the specified threshold. Specified in 8-bit notation. Valid range: 1 to 255. Default is `255`.
* `planes`: List of planes to be filtered. By default, all planes are processed.

## Clamp
`artyfox.Clamp(clip clip, clip bright_limit, clip dark_limit[, int overshoot=0, int undershoot=0, int[] planes=[0, 1, 2]])`

Analogue of `mt_clamp` from `MaskTools`. Forces the value of the `clip` to be between `bright_limit + overshoot` and `dark_limit - undershoot`. Gives unwanted results if `bright_limit + overshoot` < `dark_limit - undershoot`.
* `clip`: The clip to which the clamp is applied. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `bright_limit`: A clip that serves as a source of upper threshold for the clamp. The width, height, format and number of frames must match the original clip.
* `dark_limit`: A clip that serves as a source of lower threshold for the clamp. The width, height, format and number of frames must match the original clip.
* `overshoot`: A value that increases the upper clamp threshold by adding it to `bright_limit`. Specified in 8-bit notation. Valid range: 0 to 255. Default is `0`.
* `undershoot`: A value that decreases the lower clamp threshold by subtracting it from `dark_limit`. Specified in 8-bit notation. Valid range: 0 to 255. Default is `0`.
* `planes`: List of planes to be clamped. By default, all planes are processed.

## MakeDiff
`artyfox.MakeDiff(clip clip0, clip clip1[, int[] planes=[0, 1, 2]])`

Analogue of `mt_makediff` from `MaskTools`. On integer sample types, it behaves exactly like the standard one; on float sample types, it clamps the input and output ranges to the standard limits and aligns the brightness.
* `clip0`: The clip from which `clip1` is subtracted. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `clip1`: The clip to subtract from `clip0`. The width, height, format and number of frames must match the `clip0`.
* `planes`: List of planes to be filtered. By default, all planes are processed.

## MergeDiff
`artyfox.MergeDiff(clip clip0, clip clip1[, int[] planes=[0, 1, 2]])`

Analogue of `mt_adddiff` from `MaskTools`. On integer sample types, it behaves exactly like the standard one; on float sample types, it clamps the input and output ranges to the standard limits and aligns the brightness.
* `clip0`: The clip to which `clip1` is added. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `clip1`: The clip to add to `clip0`. The width, height, format and number of frames must match the `clip0`.
* `planes`: List of planes to be filtered. By default, all planes are processed.

## Binarize
`artyfox.Binarize(clip clip[, int thr=128, str mode='lower', int[] planes=[0, 1, 2]])`

Analogue of `mt_binarize` from `MaskTools`. It transforms soft masks into hard masks. Masks for float sample type are always in the range 0.0...1.0, regardless of plane and color family.
* `clip`: Clip for binarization. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `thr`: The threshold above which binarization occurs upwards, otherwise downwards. Specified in 8-bit notation. Valid range: 0 to 255. Default is `128`.
* `mode`: The filter's operating mode is determined by the upper and lower binarization thresholds. It works like `std.Expr('x t > a b ?')` , where `t` is the threshold and `a b` is the mode. Possible values: `255 0` (`lower`), `0 255` (`upper`), `0 x`, `x 0`, `0 t`, `t 0`, `t x`, `x t`, `255 x`, `x 255`, `255 t` and `t 255`. Default is `lower`.
* `planes`: List of planes to be binarized. By default, all planes are processed.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
