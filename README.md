# artyfox-plugin
A disjointed set of filters for VapourSynth, I write everything that seems interesting.  
The library is written using AVX2 intrinsics.
## Resize
`artyfox.Resize(clip clip, int width, int height[, float src_left=0.0, float src_top=0.0, float src_width=clip.width, float src_height=clip.height, str kernel="area", float b=1/3, float c=1/3, int taps=3, str gamma='srgb' or 'smpte170m', float sharp=1.0])`

Implementation of multiple resize functions using convolution method in a linear color space.  
Area Resize based on the publications "Algorithm and program to downsizing the digital images" by S. Z. Sverdlov and "Pixel mixing" by Jason Summers.  
Magic Kernel is based on the publication "The magic kernel" by John Costella.
* `clip`: Source clip to resize. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `width`: Target width. Must be integer and match the source clip's subsampling.
* `height`: Target height. Must be integer and match the source clip's subsampling.
* `src_left`: The `x` coordinate of the point where the region to be resized starts. Defaults to 0.0
* `src_top`: The `y` coordinate of the point where the region to be resized starts. Defaults to 0.0
* `src_width`: The width of the region to be resized relative to `src_left`. Defaults to the width of the source clip.
* `src_height`: The height of the region to be resized relative to `src_top`. Defaults to the height of the source clip.
* `kernel`: Selecting a kernel for convolution. Possible values:
  * `area`: Area Resize, used by default.
  * `bicubic`: Bicubic interpolation.
  * `bilinear`: Bilinear interpolation.
  * `blackman`: Blackman windowed sinc.
  * `gauss`: Gaussian kernel. `p` is specified via the `b` parameter, the default value is 30.
  * `kaiser`: Kaiser–Bessel windowed sinc. `beta` (`Pi` * `alpha`) is specified via the `b` parameter, the default value is 4.
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
* `taps`: Window radius value for `blackman`, `gauss`, `kaiser`, `lanczos` and `nuttall` kernels. Default is 3.
* `gamma`: The inverse and forward gamma correction value. Correction is performed before and after resizing, in order to produce the resize itself in a linear color space. The default values ​​are `'srgb'` for RGB and `'smpte170m'` for YUV and GRAY. Two different formulas are used for RGB and YUV/GRAY. The formula for YUV/GRAY is suitable for SMPTE 170M, BT.601, BT.709, BT.2020.
Other supported values ​​are: `'adobe'` (Adobe RGB), `'dcip3'` (DCI-P3), `'smpte240m'` (SMPTE 240M) and `'none'` (completely disables correction, resizing occurs directly, in a logarithmic color space).
* `sharp`: Optional post sharp. It is performed after resizing, but before gamma correction. By default, 1.0 (sharp is disabled). Values ​​​​less than 1.0 - blur, more - sharp. The allowed range of values ​​is from 0.1 to 5.0

Chroma alignment in YUV with subsampling is performed based on the `"_ChromaLocation"` property. If the property is missing or has an incorrect value, then alignment is performed along the left edge, as in MPEG2.  
Resize and alignment by fields are not supported.

## Descale
`artyfox.Descale(clip clip, int width, int height[, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, str kernel="area", float b=1/3, float c=1/3, int taps=3, float lambda=1e-4])`

Descaling via Tikhonov regularization and Cholesky decomposition (U.T @ U). This is still just an early prototype, so the fps is a bit disappointing.
* `clip`: Source clip to descale. Must be RGB, YUV or GRAY. 32-bit float sample type only.
* `width`: Target width. Must be integer and match the source clip's subsampling.
* `height`: Target height. Must be integer and match the source clip's subsampling.
* `src_left`: The `x` coordinate of the point where the destination region after descaling starts. Defaults to 0.0
* `src_top`: The `y` coordinate of the point where the destination region after descaling starts. Defaults to 0.0
* `src_width`: The width of the destination region after descaling relative to `src_left`. Defaults to `width`.
* `src_height`: The height of the destination region after descaling relative to `src_top`. Defaults to `height`.
* `kernel`: Selecting a kernel for deconvolution. See `Resize` for possible values.
* `b`: The `b` parameter in the `bicubic` kernel. Defaults to 1/3.
* `c`: The `c` parameter in the `bicubic` kernel. Defaults to 1/3.
* `taps`: Window radius value for `blackman`, `gauss`, `kaiser`, `lanczos` and `nuttall` kernels. Default is 3.
* `lambda`: Regularization parameter. Ensures positive definiteness and stability of the solution to the system of equations. Small values ​​can lead to increased noise, quantization artifacts, and ringing. Excessively large values ​​produce a smooth image, suppressing fine details. Default is 1e-4. Valid range: 1e-16 <= `lambda` < 1. Since `lambda` is a Python keyword, you may [append an underscore to the argument’s name when invoking the filter](https://www.vapoursynth.com/doc/pythonreference.html#python-keywords-as-filter-arguments).

## RelativeError
`artyfox.RelativeError(clip clip0, clip clip1)`

Calculates the relative error of the two input clips and stores it as the `"RelativeError"` property of the first clip.
* `clip0`: Original clip. Must be GRAY. 32-bit float sample type only.
* `clip1`: Restored clip. Must be GRAY. 32-bit float sample type only. The width, height and number of frames must match the original clip.

## Linearize
`artyfox.Linearize(clip clip[, str gamma='srgb' or 'smpte170m', int[] planes=[0, 1, 2]])`

Inverse gamma correction (linearization) of the color space.
* `clip`: Source clip to linearize. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: The inverse and forward gamma correction value. Correction is performed before and after resizing, in order to produce the resize itself in a linear color space. The default values ​​are `'srgb'` for RGB and `'smpte170m'` for YUV and GRAY. Two different formulas are used for RGB and YUV/GRAY. The formula for YUV/GRAY is suitable for SMPTE 170M, BT.601, BT.709, BT.2020.
Other supported values ​​are: `'adobe'` (Adobe RGB), `'dcip3'` (DCI-P3) and `'smpte240m'` (SMPTE 240M).
* `planes`: List of planes to linearize. Default is all.

## GammaCorr
`artyfox.GammaCorr(clip clip[, str gamma='srgb' or 'smpte170m', int[] planes=[0, 1, 2]])`

Gamma correction of color space.
* `clip`: Source clip for gamma correction. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: The inverse and forward gamma correction value. Correction is performed before and after resizing, in order to produce the resize itself in a linear color space. The default values ​​are `'srgb'` for RGB and `'smpte170m'` for YUV and GRAY. Two different formulas are used for RGB and YUV/GRAY. The formula for YUV/GRAY is suitable for SMPTE 170M, BT.601, BT.709, BT.2020.
Other supported values ​​are: `'adobe'` (Adobe RGB), `'dcip3'` (DCI-P3) and `'smpte240m'` (SMPTE 240M).
* `planes`: List of planes to be gamma corrected. Default is all.

## BitDepth
`artyfox.BitDepth(clip clip, int bits[, bool direct=False])`

Converting the bit depth of a clip.
* `clip`: Source clip to be converted to bit depth. Must be RGB, YUV or GRAY. 8-16-bit integer or 32-bit float sample type.
* `bits`: The bit depth of the target clip. It can be from `8` to `16` or `32`. When converting from integer to float or vice versa, a color range conversion may also occur, since in the 32-bit float format, the concept of a limited range does not exist. The range is converted according to the frame's `"_ColorRange"` property. If this property does not exist or has an invalid value, the range is considered full for RGB and limited for YUV and GRAY. Conversion between integers occurs without regard to range. Downconversion of bit depth occurs with arithmetic rounding and saturation.
* `direct`: If `True`, conversion from integer to float or vice versa always uses the full range and ignores the `"_ColorRange"` property. Defaults to `False`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
