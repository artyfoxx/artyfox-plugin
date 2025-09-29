# artyfox-plugin
A disjointed set of filters for VapourSynth, I write everything that seems interesting.  
The author is a lover of long sheets of code stuffed into a single file. Don't judge too harshly.  
Only for processors with AVX2 and FMA support!
## Resize
`artyfox.Resize(clip clip, int width, int height[, float src_left = 0.0, float src_top = 0.0, float src_width = clip.width, float src_height = clip.height, str kernel = "area", float b = 1/3, float c = 1/3, int taps = 3, float gamma = 2.4 or ≈2.2, float sharp = 1.0])`

Implementation of multiple resize functions using convolution method.  
Area Resize based on the publications "Algorithm and program to downsizing the digital images" by S. Z. Sverdlov and "Pixel mixing" by Jason Summers.  
Magic Kernel is based on the publication "The magic kernel" by John Costella.
* `clip`: Source clip to downscale. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
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
* `taps`: Window radius value for `blackman`, `lanczos`, and `nuttall` kernels. Default is 3.
* `gamma`: The inverse and forward gamma correction value. Correction is performed before and after resizing, in order to produce the resize itself in a linear color space. The default values ​​are 2.4 for RGB and ≈2.2 (1 / 0.45) for YUV and GRAY. Two different formulas are used for RGB and YUV/GRAY. The formula for YUV/GRAY is suitable for SMPTE 170M, BT.601, BT.709, BT.2020 and is not suitable for SMPTE 240M, DCI-P3.  
`gamma = 1` - completely disables correction, resizing occurs directly, in a logarithmic color space.  
The allowed range of values ​​is from 0.1 to 5.0
* `sharp`: Optional post sharp. It is performed after resizing, but before gamma correction. By default, 1.0 (sharp is disabled). Values ​​​​less than 1.0 - blur, more - sharp. The allowed range of values ​​is from 0.1 to 5.0

Chroma alignment in YUV with subsampling is performed based on the `"_ChromaLocation"` property. If the property is missing or has an incorrect value, then alignment is performed along the left edge, as in MPEG2.  
Resize and alignment by fields are not supported.

## Linearize
`artyfox.Linearize(clip clip[,float gamma = 2.4 or ≈2.2, int[] planes=[0, 1, 2]])`

Inverse gamma correction (linearization) of the color space.
* `clip`: Source clip to linearize. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: Inverse gamma correction value. The default values ​​are 2.4 for RGB and ≈2.2 (1 / 0.45) for YUV and GRAY. Two different formulas are used for RGB and YUV/GRAY. The formula for YUV/GRAY is suitable for SMPTE 170M, BT.601, BT.709, BT.2020 and is not suitable for SMPTE 240M, DCI-P3. The allowed range of values ​​is from 0.1 to 5.0
* `planes`: List of planes to linearize. Default is all.

## GammaCorr
`artyfox.GammaCorr(clip clip[,float gamma = 2.4 or ≈2.2, int[] planes=[0, 1, 2]])`

Gamma correction of color space.
* `clip`: Source clip for gamma correction. Must be RGB, YUV or GRAY. 32-bit float sample type only. The range must be converted to full.
* `gamma`: Gamma correction value. The default values ​​are 2.4 for RGB and ≈2.2 (1 / 0.45) for YUV and GRAY. Two different formulas are used for RGB and YUV/GRAY. The formula for YUV/GRAY is suitable for SMPTE 170M, BT.601, BT.709, BT.2020 and is not suitable for SMPTE 240M, DCI-P3. The allowed range of values ​​is from 0.1 to 5.0
* `planes`: List of planes to be gamma corrected. Default is all.
## To do
Maybe add a couple more interesting convolution kernels.
