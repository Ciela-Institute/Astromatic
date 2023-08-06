import torch
import torch.nn.functional as F
from torch.func import vmap
from astropy.wcs import WCS
from astropy import units
from astropy.io import fits
from astropy.coordinates import SkyCoord
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.restoration import richardson_lucy

def deconvolve_fft(image, psf, pad_size = 100):
    """
    Deconvolve a PSF from an image using FFT and return the sharpened image.

    image: 2d numpy array image
    psf: 2d numpy image with centered point source. Must have odd number of pixels on each side. Point source is centered at the middle of the central pixel.
    """

    # Ensure PSF has odd number of pixels because that's easier
    assert psf.shape[0] % 2 != 0, "psf image must have odd number of pixels"
    assert psf.shape[1] % 2 != 0, "psf image must have odd number of pixels"
    
    image = np.pad(image, pad_width = pad_size, mode = 'edge')
    
    # Convert image and psf to frequency space
    image_fft = fftshift(fft2(image))
    psf_fft = fftshift(fft2(psf, image.shape))
    
    deconvolved_fft = np.clip(image_fft / psf_fft, a_min=0., a_max=None)
    cut_freq = int(image.shape[0]/2 - 5*psf.shape[0])
    smooth_fft = np.zeros(deconvolved_fft.shape, dtype=np.cdouble)
    smooth_fft[cut_freq:-cut_freq,cut_freq:-cut_freq] = deconvolved_fft[cut_freq:-cut_freq,cut_freq:-cut_freq]
    # Return real space deconvolved image
    deconvolved_image = np.abs(ifft2(ifftshift(smooth_fft)))

    return deconvolved_image[pad_size:-pad_size,pad_size:-pad_size]
    

def deconvolve_lucyrichardson(image, psf, n_iter=30, pad_size = 100, filter_epsilon=None):
    """
    Deconvolve a PSF from an image using the Lucy-Richardson algorithm and return the sharpened image.

    image: 2d numpy array image
    psf: 2d numpy image with centered point source. Must have odd number of pixels on each side. Point source is centered at the middle of the central pixel.
    n_iter: number of Lucy-Richardson iterations to perform.
    """

    # Ensure PSF has odd number of pixels because that's easier
    assert psf.shape[0] % 2 != 0, "psf image must have odd number of pixels"
    assert psf.shape[1] % 2 != 0, "psf image must have odd number of pixels"

    # Record pixel flux limits from image. These are used to scale to the -1,1 range
    dmax = np.max(image)
    dmin = np.min(image)
    
    image = np.pad(image, pad_width = pad_size, mode = 'wrap')
    
    # Perform the LR deconvolution on the scaled image
    deconv = richardson_lucy(
        2 * (image - dmin) / (dmax - dmin) - 1,
        psf,
        num_iter=n_iter,
        filter_epsilon=filter_epsilon
    )
    deconv = (deconv + 1) * ((dmax - dmin) / 2) + dmin
    # Rescale back to the original flux range and return
    return deconv[pad_size:-pad_size,pad_size:-pad_size]


def interpolate(image, coordinates):
    """
    Interpolation function, without a batch size. To make it batched, used vmap from functorch.
    """
    C, H, W = image.shape
    x, y = torch.tensor_split(coordinates, 2, dim=0)
    x = x.view(-1)
    y = y.view(-1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clip(x0, 0, W - 1)
    x1 = torch.clip(x1, 0, W - 1)
    y0 = torch.clip(y0, 0, H - 1)
    y1 = torch.clip(y1, 0, H - 1)
    x = torch.clip(x, 0, W - 1)
    y = torch.clip(y, 0, H - 1)

    Ia = image[..., x0, y0]
    Ib = image[..., x0, y1]
    Ic = image[..., x1, y0]
    Id = image[..., x1, y1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    _, new_H, new_W = coordinates.shape
    return (wa * Ia + wb * Ib + wc * Ic + wd * Id).view(C, new_H, new_W)


def make_wcs(skycoord, orientation, pixels, pixel_size):
    """
    Create a World Coordinate System (WCS) object based on the given parameters.

    Parameters:
    skycoord (SkyCoord): The sky coordinates of the center of the image.
    orientation (float): The orientation of the image, defined as East of North in world coordinates.
    pixels (int): The number of pixels in each dimension of the image.
    pixel_size (Quantity): The size of each pixel in angular units.

    Returns:
    WCS: The World Coordinate System object.

    Raises:
    None.

    This function creates a FITS header and populates it with the necessary keywords to define a WCS.
    The FITS header is then used to create a WCS object, which can be used to convert between pixel coordinates and sky coordinates.

    The FITS header is populated with the following keywords:
    - NAXIS1: The number of pixels in the x-axis of the image.
    - NAXIS2: The number of pixels in the y-axis of the image.
    - CRVAL1: The right ascension of the center of the image in degrees.
    - CRVAL2: The declination of the center of the image in degrees.
    - CUNIT1: The units of the x-axis coordinates (degrees).
    - CUNIT2: The units of the y-axis coordinates (degrees).
    - CTYPE1: The coordinate type of the x-axis (RA---TAN).
    - CTYPE2: The coordinate type of the y-axis (DEC--TAN).
    - PCi_j: Elements of the pixel scale matrix for converting from pixel coordinates to sky coordinates.

    The orientation of the image is defined as East of North in world coordinates.
    To convert the orientation to pixel coordinate rotation, a 90 degree rotation is added.
    The rotation matrix is then calculated using the pixel size and the rotated orientation.
    The rotation matrix accounts for the mirror flip of the y-axis pixel coordinate (East<-West).

    The WCS object is created using the FITS header and returned.
    """
    hdr = fits.Header()
    hdr["NAXIS1"] = pixels
    hdr["NAXIS2"] = pixels
    hdr["CRVAL1"] = skycoord.ra.to(units.deg).value
    hdr["CRVAL2"] = skycoord.dec.to(units.deg).value
    hdr["CRPIX1"] = pixels/2 - 0.5 # Same convention as Cutout2D
    hdr["CRPIX2"] = pixels/2 - 0.5
    hdr["CUNIT1"] = 'deg'
    hdr["CUNIT2"] = 'deg'
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CDELT1"] = pixel_size.to(units.deg).value
    hdr["CDELT2"] = pixel_size.to(units.deg).value
    theta = orientation * np.pi / 180
    rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta), np.cos(theta)]])
    mirror_j = np.array([[1, 0], [0, -1]])
    pc = rotation @ mirror_j
    hdr["PC1_1"] = pc[0, 0]
    hdr["PC1_2"] = pc[0, 1]
    hdr["PC2_1"] = pc[1, 0]
    hdr["PC2_2"] = pc[1, 1]
    return WCS(hdr)
    

def make_forward_model(
        psf:np.ndarray, 
        model_pixels:int,
        model_pixel_size:float,
        observation_pixels:int,
        observation_pixel_size:float,
        wcs_list:list[WCS, ...]=None, 
        super_sampling_factor:int=4,
        zero_padding:int=0,
        fiducial_center:SkyCoord=None,
        fiducial_orientation:float=None, # Pick the orientation of the first WCS, angle East of North
        **kwargs
        ):
    """
    Create a forward model for a given point spread function (PSF) and a list of world coordinate systems (WCS).

    Parameters:
    -----------
    psf : np.ndarray
        The point spread function (PSF) to be used for the forward model. It should be a 2D or 3D array (multi channel fit).

    wcs_list : list[WCS, ...]
        A list of world coordinate systems (astropy WCS) to be used for the forward model. 

    super_sampling_factor : int
        The super sampling factor to be used for the forward model. It determines the level of detail in the model.

    model_pixels : int
        The number of pixels to be used for the model pixel grid.

    model_pixel_size : units.Quantity
        The size of each pixel in the model pixel grid. It should be an instance of the units.Quantity class.

    zero_padding : int, optional
        The number of zero padding pixels to be added to the model pixel grid on each side. Default is 0.

    fiducial_center : SkyCoord, optional
        The fiducial center to be used for the model reference pixel. It should be an instance of the SkyCoord class. 
        Default (None) is to use first WCS center pixel world coordinate.
        
    fiducial_orientation : float, optional
        The fiducial orientation to be used for model pixel grid. The angle is defined East of North. 
        Default (None) is to use first WCS pixel scale matrix orientation.

    Returns:
    --------
    forward_model : np.ndarray
        The created forward model as a 2D array.

    Examples:
    ---------
    >>> psf = np.ones((5, 5))
    >>> wcs_list = [WCS(), WCS()]
    >>> super_sampling_factor = 2
    >>> model_pixels = 10
    >>> model_pixel_size = 0.1
    >>> forward_model = make_forward_model(psf, wcs_list, super_sampling_factor, model_pixels, model_pixel_size)
    """
    if wcs_list is None:
        center = SkyCoord(ra=10*units.deg, dec=20*units.deg)
        wcs_list = [make_wcs(center, 0., pixels=observation_pixels, pixel_size=observation_pixel_size * units.arcsec)]
    C = 1
    H, W = psf.shape
    psf = torch.tensor(psf).float().to(DEVICE).view(C, 1, H, W) # reshape to a convolution kernel [channel_out, channels_in/groups, H, W]
    batched_interpolation = vmap(interpolate, in_dims=(0, None))  # only batch over the images (first argument of interpolate)
    
    if fiducial_center is None:
        # Use same convention as Cutout2D for reference pixel
        center = [dim / 2 - 0.5 for dim in wcs_list[0].pixel_shape]
        fiducial_center = wcs_list[0].pixel_to_world(*center)
    if fiducial_orientation is None:
        pc = wcs_list[0].pixel_scale_matrix
        fiducial_orientation = np.arctan2(pc[1, 0], pc[0, 0]) * 180 / np.pi
    fiducial_wcs = make_wcs(fiducial_center, fiducial_orientation, model_pixels, model_pixel_size * units.arcsec)

    # Prepare coordinate systems
    model_coordinates_list = []
    for wcs in wcs_list:
        # Observation pixel coordinates super sampled
        u = (np.arange(super_sampling_factor * wcs.pixel_shape[0]) + 0.5) / super_sampling_factor 
        v = (np.arange(super_sampling_factor * wcs.pixel_shape[1]) + 0.5) / super_sampling_factor 
        # v = np.flip(v) # Remember matrix convention for pixel indexing
        u, v = np.meshgrid(u, v, indexing="ij")
        world = wcs.pixel_to_world(u, v)
        model_coordinates = np.stack(fiducial_wcs.world_to_pixel(world), axis=0)
        model_coordinates_list.append(torch.tensor(model_coordinates).float().to(DEVICE))
    
    def A(x):
        x = F.pad(x, pad=[zero_padding]*4, mode="constant", value=0.)
        ys = []
        for i in range(len(wcs_list)):
            y = batched_interpolation(x, model_coordinates_list[i])
            y = F.conv2d(y, psf, groups=C, padding="same")
            y = F.avg_pool2d(y, kernel_size=super_sampling_factor, stride=super_sampling_factor)
            ys.append(y)
        return torch.concat(ys, dim=1)
    return A


