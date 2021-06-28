# Licensed under a 3-clause BSD style license - see LICENSE.rst

# imagecube
# This package accepts FITS images from the user and delivers images that have
# been converted to the same flux units, registered to a common world
# coordinate system (WCS), convolved to a common resolution, and resampled to a
# common pixel scale requesting the Nyquist sampling rate.
# Each step can be run separately or as a whole.
# The user should provide us with information regarding wavelength, pixel
# scale extension of the cube, instrument, physical size of the target, and WCS
# header information.

from __future__ import division, print_function

import getopt
import glob
import gzip
import math
import os
import shutil
import sys
import warnings
from datetime import datetime

from astropy import constants
from astropy import log
from astropy import units as u
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from astropy.io import fits
import astropy.utils.console as console
from astropy.utils.exceptions import AstropyUserWarning

# import the file containing all the constants
from imagecube.imagecube_constants import (FUV_LAMBDA_CON,
                                           FVEGA_H,
                                           FVEGA_J,
                                           FVEGA_KS,
                                           JY_CONVERSION,
                                           MJY_PER_SR_TO_JY_PER_ARCSEC2,
                                           NUV_LAMBDA_CON,
                                           NYQUIST_SAMPLING_RATE,
                                           S250_BEAM_AREA,
                                           S350_BEAM_AREA,
                                           S500_BEAM_AREA, )

# also import WAVELENGTH_2MASS_H, WAVELENGTH_2MASS_J, WAVELENGTH_2MASS_KS

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

import montage_wrapper as montage

import numpy as np

import requests


matplotlib.use('PS')


def print_usage():
    """
    Displays usage information in case of a command line error.
    """

    print("""
Usage: """ + sys.argv[0] + """ --dir <directory> --ang_size <angular_size>
[--flux_conv] [--im_reg] [--im_ref <filename>] [--rot_angle <number in degree>]
[--im_conv] [--fwhm <fwhm value>] [--kernels <kernel directory>] [--im_regrid]
[--im_pixsc <number in arcsec>] [--seds] [--make2d] [--cleanup] [--help]

dir: the path to the directory containing the <input FITS files> to be
processed. For multi-extension FITS files, currently only the first extension
after the primary one is used.

ang_size: the field of view of the output image cube in arcsec

flux_conv: perform unit conversion to Jy/pixel for all images not already
in these units.
NOTE: If data are not GALEX, 2MASS, MIPS, IRAC, PACS, SPIRE, then the user
should provide flux unit conversion factors to go from the image's native
flux units to Jy/pixel. This information should be recorded in the header
keyword FLUXCONV for each input image.

im_reg: register the input images to the reference image. The user should
provide the reference image with the im_ref parameter.

im_ref: user-provided reference image to which the other images are registered.
This image must have a valid world coordinate system. The position angle of
thie image will be used for the final registered images, unless an
angle is explicitly set using --rot_angle.

rot_angle: position angle (+y axis, in degrees West of North) for the
registered images.
If omitted, the PA of the reference image is used.

im_conv: perform convolution to a common resolution, using either a Gaussian or
a PSF kernel. For Gaussian kernels, the angular resolution is specified
with the fwhm parameter. If the PSF kernel is chosen, the user provides the
PSF kernels with the following naming convention:

    <input FITS files>_kernel.fits

For example: an input image named SI1.fits will have a corresponding
kernel file named SI1_kernel.fits

fwhm: the angular resolution in arcsec to which all images will be convolved
with im_conv, if the Gaussian convolution is chosen, or if not all the input
images have a corresponding kernel.

kernels: the name of a directory containing kernel FITS
images for each of the input images. If all input images do not have a
corresponding kernel image, then the Gaussian convolution will be performed for
these images.

im_regrid: perform regridding of the convolved images to a common
pixel scale. The pixel scale is defined by the im_pxsc parameter.

im_pixsc: the common pixel scale (in arcsec) used for the regridding
of the images in the im_regrid. It is a good idea the pixel scale and angular
resolution of the images in the regrid step to conform to the Nyquist sampling
rate: angular resolution = """ + str(NYQUIST_SAMPLING_RATE) + """ * im_pixsc

seds:  produce the spectral energy distribution on a pixel-by-pixel
basis, on the regridded images.

make2d: along with the true 3D datacube to be built, create a multi extension
file, stored with a _2d appended to the datacube filename

cleanup: if this parameter is present, then output files from previous
executions of the script are removed and no processing is done.

help: if this parameter is present, this message will be displayed and no
processing will be done.

NOTE: the following keywords must be present in all images, along with a
comment containing the units (where applicable), for optimal image processing:

    BUNIT: the physical units of the array values (i.e. the flux unit).
    FLSCALE: the factor that converts the native flux units (as given
             in the BUNIT keyword) to Jy/pixel. The units of this factor should
             be: (Jy/pixel) / (BUNIT unit). This keyword should be added in the
             case of data other than GALEX (FUV, NUV), 2MASS (J, H, Ks),
             SPITZER (IRAC, MIPS), HERSCHEL (PACS, SPIRE; photometry)
    INSTRUME: the name of the instrument used
    WAVELNTH: the representative wavelength (in micrometres) of the filter
              bandpass
Keywords which constitute a valid world coordinate system must also be present.

If any of these keywords are missing, imagecube will attempt to determine them.
The calculated values will be present in the headers of the output images;
if they are not the desired values, please check the headers
of your input images and try again.
    """)


def parse_command_line(args):
    """
    Parses the command line to obtain parameters.

    """

    global ang_size
    global image_directory
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global main_reference_image
    global fwhm_input
    global kernel_directory
    global im_pixsc
    global rot_angle
    global make_2D

    # switch to argparse
    parse_status = 0
    try:
        opts, args = getopt.getopt(args, "", ["dir=", "ang_size=", "flux_conv",
                                              "im_conv", "im_reg", "im_ref=",
                                              "rot_angle=", "im_conv",
                                              "fwhm=", "kernels=",
                                              "im_pixsc=", "im_regrid",
                                              "seds", "cleanup",
                                              "help", "make2d"])
    except getopt.GetoptError as exc:
        print(exc.msg)
        print("An error occurred. Check your parameters and try again.")
        parse_status = 2
        return(parse_status)
    for opt, arg in opts:
        if opt in ("--help"):
            print_usage()
            parse_status = 1
            return(parse_status)
        elif opt in ("--ang_size"):
            ang_size = float(arg)
        elif opt in ("--dir"):
            image_directory = arg
            if (not os.path.isdir(image_directory)):
                print("Error: The directory cannot be found: " +
                      image_directory)
                parse_status = 2
                return(parse_status)
        elif opt in ("--flux_conv"):
            do_conversion = True
        elif opt in ("--im_reg"):
            do_registration = True
        elif opt in ("--rot_angle"):
            rot_angle = float(arg)
        elif opt in ("--im_conv"):
            do_convolution = True
        elif opt in ("--im_regrid"):
            do_resampling = True
        elif opt in ("--seds"):
            do_seds = True
        elif opt in ("--cleanup"):
            do_cleanup = True
        elif opt in ("--im_ref"):
            main_reference_image = arg
        elif opt in ("--fwhm"):
            fwhm_input = float(arg)
        elif opt in ("--make2d"):
            make_2D = True
        elif opt in ("--kernels"):
            kernel_directory = arg
            if (not os.path.isdir(kernel_directory)):
                print("Error: The directory cannot be found: " +
                      kernel_directory)
                parse_status = 2
                return
        elif opt in ("--im_pixsc"):
            im_pixsc = float(arg)

    if (main_reference_image != ''):
        try:
            with open(main_reference_image):
                pass
        except IOError:
            print("The file " + main_reference_image +
                  " could not be found in the directory " + image_directory +
                  ". Cannot run without reference image, exiting.")
            parse_status = 2
    return(parse_status)


def get_conversion_factor(header, instrument):
    """
    Returns the factor that is necessary to convert an image's native "flux
    units" to Jy/pixel.

    Parameters
    ----------
    header: FITS file header
        The header of the FITS file to be checked.

    instrument: string
        The instrument which the data in the FITS file came from

    Returns
    -------
    conversion_factor: float
        The conversion factor that will convert the image's native "flux
        units" to Jy/pixel.
    """

    # Give a default value that can't possibly be valid; if this is still the
    # value after running through all of the possible cases, then an error has
    # occurred.
    conversion_factor = 0
    pixelscale = get_pixel_scale(header)

    if (instrument == 'IRAC'):
        conversion_factor = (MJY_PER_SR_TO_JY_PER_ARCSEC2) * (pixelscale**2)

    elif (instrument == 'MIPS'):
        conversion_factor = (MJY_PER_SR_TO_JY_PER_ARCSEC2) * (pixelscale**2)

    elif (instrument == 'GALEX'):
        # there seems to be a different name for wavelength in some images,
        # look into it
        wavelength = u.um.to(u.angstrom, float(header['WAVELNTH']))
        f_lambda_con = 0
        # I am using a < comparison here to account for the possibility that
        # the given wavelength is not EXACTLY 1520 AA or 2310 AA
        if (wavelength < 2000):
            f_lambda_con = FUV_LAMBDA_CON
        else:
            f_lambda_con = NUV_LAMBDA_CON
        conversion_factor = (((JY_CONVERSION) * f_lambda_con * wavelength**2) /
                             (constants.c.to('angstrom/s').value))

    elif (instrument == '2MASS'):
        fvega = 0
        if (header['FILTER'] == 'j'):
            fvega = FVEGA_J
        elif (header['FILTER'] == 'h'):
            fvega = FVEGA_H
        elif (header['FILTER'] == 'k'):
            fvega = FVEGA_KS
        conversion_factor = fvega * 10**(-0.4 * header['MAGZP'])

    elif (instrument == 'PACS'):
        # Confirm that the data is already in Jy/pixel by checking the BUNIT
        # header keyword
        if ('BUNIT' in header):
            if (header['BUNIT'].lower() != 'jy/pixel'):
                log.info("Instrument is PACS, but Jy/pixel " +
                         "is not being usedin " + "BUNIT.")
        conversion_factor = 1

    elif (instrument == 'SPIRE'):
        wavelength = float(header['WAVELNTH'])
        if (wavelength == 250):
            conversion_factor = (pixelscale**2) / S250_BEAM_AREA
        elif (wavelength == 350):
            conversion_factor = (pixelscale**2) / S350_BEAM_AREA
        elif (wavelength == 500):
            conversion_factor = (pixelscale**2) / S500_BEAM_AREA

    return conversion_factor


def convert_images(image_stack):
    """
    Converts all of the input images' native "flux units" to Jy/pixel
    The converted values are stored in the list of arrays,
    converted_data, and they are also saved as new FITS images.

    Parameters
    ----------
    image_stack: HDU list
        A structure containing headers and image data for all FITS input
        images.


    """
    # make new directory for output, if needed
    new_directory = image_directory + "/converted/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    for i in range(1, len(image_stack)):
        if ('FLSCALE' in image_stack[i].header):
            conversion_factor = float(image_stack[i].header['FLSCALE'])
        else:
            try:  # try to get conversion factor from image header
                instrument = image_stack[i].header['INSTRUME']
                conversion_factor = get_conversion_factor(
                    image_stack[i].header, instrument)
            except KeyError:  # get this if no 'INSTRUME' keyword
                conversion_factor = 0
            # if conversion_factor == 0 either we don't know the instrument
            # or we don't have a conversion factor for it.
            if conversion_factor == 0:
                warnings.warn("No conversion factor for image %s, using 1"
                              % image_stack[i].header['FILENAME'],
                              AstropyUserWarning)
                conversion_factor = 1.0

        # Some manipulation of filenames and directories
        original_filename = os.path.basename(image_stack[i].header['FILENAME'])
        converted_filename = (new_directory + original_filename +
                              "_converted.fits")

        # Do a Jy/pixel unit conversion and save it as a new .fits file
        image_stack[i].data = image_stack[i].data * conversion_factor
        converted_data.append(image_stack[i].data)
        image_stack[i].header['BUNIT'] = 'Jy/pixel'
        image_stack[i].header['JYPXFACT'] = (
            conversion_factor, 'Factor to' +
            ' convert original BUNIT into Jy/pixel.'
        )

        hdu = fits.PrimaryHDU(image_stack[i].data, image_stack[i].header)
        hdu.writeto(converted_filename, overwrite=True)
    return image_stack

# modified from aplpy.wcs_util.get_pixel_scales


def get_pixel_scale(header):
    '''
    Compute the pixel scale in arcseconds per pixel from an image WCS
    Assumes WCS is in degrees

    Parameters
    ----------
    header: FITS header of image

    Returns
    -------
    pix_scale: float
        Pixel scale in arcseconds per pixel


    '''
    w = wcs.WCS(header)

    if w.wcs.has_cd():
        # get_cdelt is supposed to work whether header has CDij, PC, or CDELT
        pc = np.matrix(w.wcs.get_pc())
        pix_scale = math.sqrt(pc[0, 0]**2 + pc[0, 1]**2) * u.deg.to(u.arcsec)
    else:  # but don't think it does
        pix_scale = abs(w.wcs.get_cdelt()[0]) * u.deg.to(u.arcsec)
    return(pix_scale)


def get_pangle(header):
    '''
    Compute the rotation angle, in degrees,  from an image WCS
    Assumes WCS is in degrees

    Parameters
    ----------
    header: FITS header of image

    Returns
    -------
    cr2: float
        Rotation angle in degrees

    '''
    w = wcs.WCS(header)
    pc = w.wcs.get_pc()
    cr2 = math.atan2(pc[0, 1], pc[0, 0]) * u.radian.to(u.deg)
    return(cr2)


def merge_headers(montage_hfile, orig_header, out_file):
    '''
    Merges an original image header with the WCS info
    in a header file generated by montage.mHdr.
    Puts the results into out_file.


    Parameters
    ----------
    montage_hfile: a text file generated by montage.mHdr,
    which contains only WCS information
    orig_header: FITS header of image, contains all the other
    stuff we want to keep

    '''
    montage_header = fits.Header.fromtextfile(montage_hfile)
    for key in orig_header.keys():
        if key in montage_header.keys():
            # overwrite the original header WCS
            orig_header[key] = montage_header[key]
    if 'CD1_1' in orig_header.keys():
        # if original header has CD matrix instead of CDELTs:
        for cdm in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
            if cdm in orig_header.keys():
                del orig_header[cdm]  # delete the CD matrix
        for cdp in ['CDELT1', 'CDELT2', 'CROTA2']:
            # insert the CDELTs and CROTA2
            orig_header[cdp] = montage_header[cdp]
    orig_header.tofile(out_file, sep='\n', endcard=True,
                       padding=False, overwrite=True)
    return orig_header


def get_ref_wcs(img_name):
    '''
    get WCS parameters from first science extension
    (or primary extension if there is only one) of image

    Parameters
    ----------
    img_name: name of FITS image file


    '''
    hdulist = fits.open(img_name)
    # take the first sci image if multi-ext.
    hdr = hdulist[find_image_planes(hdulist)[0]].header
    lngref_input = hdr['CRVAL1']
    latref_input = hdr['CRVAL2']
    try:
        rotation_pa = rot_angle  # the user-input PA
    except NameError:  # user didn't define it
        log.info('Getting position angle from %s' % img_name)
        rotation_pa = get_pangle(hdr)
    log.info('Using PA of %.1f degrees' % rotation_pa)
    hdulist.close()
    return(lngref_input, latref_input, rotation_pa)


def find_image_planes(hdulist):
    """
    Reads FITS hdulist to figure out which ones contain science data

    Parameters
    ----------
    hdulist: FITS hdulist

    Outputs
    -------
    img_plns: list of which indices in hdulist correspond to science data

    """
    n_hdu = len(hdulist)
    img_plns = []
    if n_hdu == 1:  # if there is only one extension, then use that
        img_plns.append(0)
    else:  # loop over all the extensions & try to find the right ones
        for extn in range(1, n_hdu):
            try:  # look for 'EXTNAME' keyword, see if it's 'SCI'
                if 'SCI' or 'WSAIMAGE' or 'BINTABLE' in hdulist[extn].header['EXTNAME']: #SAM ADDED WSANAME AND BINTABLE
                    img_plns.append(extn)
            except KeyError:  # no 'EXTNAME', we assume we want this extension
                img_plns.append(extn)
    return(img_plns)


def register_images(image_stack):
    """
    Registers all of the images to a common WCS

    Parameters
    ----------
    image_stack: HDU list
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/registered/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # get WCS info for the reference image
    lngref_input, latref_input, rotation_pa = get_ref_wcs(main_reference_image)
    width_and_height = (ang_size * u.arcsec).to(u.deg)/u.deg #converts to unitless so that montage can read it
    """print("---------SIZE------------")
    print(lngref_input, latref_input)
    print(width_and_height)"""

    # temporary directory to store the file from image_stack
    # so that reproject function works
    tmp_directory = image_directory + "/temp/"
    if not os.path.exists(tmp_directory):
        os.mkdir(tmp_directory)

    # now loop over all the images
    for i in range(1, len(image_stack)):

        native_pixelscale = get_pixel_scale(image_stack[i].header)

        original_filename = os.path.basename(image_stack[i].header['FILENAME'])
        artificial_filename = (new_directory + original_filename +
                               "_pixelgrid_header")
        registered_filename = (new_directory + original_filename +
                               "_registered.fits")

        # temporary file created from image_stack
        tmp_filename = (tmp_directory + original_filename)

        hdulist = fits.HDUList()
        hdu_header = image_stack[i].header
        hdu_data = image_stack[i].data
        hdulist.append(fits.PrimaryHDU(header=hdu_header, data=hdu_data))
        hdulist.writeto(tmp_filename, overwrite=True, output_verify='ignore')

        """#(SAM) SET EQUINOX TO THE EQUINOX THAT IS DEFINED IN THE HEADER
        if image_stack[i].header['EQUINOX']:
            equinox = image_stack[i].header['EQUINOX']
        else:
            equinox = 2000"""

        # make the new header & merge it with old
        montage.commands.mHdr(str(lngref_input) + ' ' + str(latref_input),
                              width_and_height, artificial_filename,
                              system='eq', equinox=2000,
                              height=width_and_height,
                              pix_size=native_pixelscale, rotation=rotation_pa)
        image_stack[i].header = merge_headers(artificial_filename,
                                              image_stack[i].header,
                                              artificial_filename)

        # reproject using montage
        montage.wrappers.reproject(tmp_filename, registered_filename,
                                   header=artificial_filename, exact_size=True)
        # delete the file with header info
        os.unlink(artificial_filename)
        # delete the temporary file made
        os.unlink(tmp_filename)
        image_stack[i].header = fits.open(registered_filename)[0].header
        image_stack[i].data = fits.open(registered_filename)[0].data

    # remove the tmeporary directory created
    os.rmdir(tmp_directory)

    return image_stack


def convolve_images(image_stack, kernel_stack):
    """
    Convolves all of the images to a common resolution using a simple
    gaussian kernel.

    Parameters
    ----------
    image_stack: HDU list
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/convolved/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    for i in range(1, len(image_stack)):
        original_filename = os.path.basename(image_stack[i].header['FILENAME'])
        original_directory = os.path.dirname(image_stack[i].header['FILENAME'])
        convolved_filename = (new_directory + original_filename +
                              "_convolved.fits")

        kernel_filename = (original_directory + "/" + kernel_directory + "/" +
                           original_filename + "_kernel.fits")

        log.info("Looking for " + kernel_filename)

        if os.path.exists(kernel_filename):
            log.info("Found a kernel")

            convolved_image = convolve_fft(image_stack[i].data,
                                           kernel_stack[i])
            hdu = fits.PrimaryHDU(convolved_image, image_stack[i].header)
            hdu.writeto(convolved_filename, overwrite=True)
            image_stack[i].data = convolved_image

        else:  # no kernel

            # NOTETOSELF: there has been a loss of data from the data cubes at
            # an earlier step. The presence of 'EXTEND' and 'DSETS___' keywords
            # in the header no longer means that there is any data in
            # hdulist[1].data. I am using a workaround for now, but this needs
            # to be looked at.
            # NOTE_FROM_PB: can possibly solve this issue, and eliminate a lot
            # of repetitive code, by making a multi-extension FITS file
            # in the initial step, and iterating over extensions in that file

            hdulist = image_stack[i]
            header = hdulist.header
            image_data = hdulist.data

            # NOTETOSELF: not completely clear whether Gaussian2DKernel 'width'
            # is sigma or FWHM also, previous version had kernel being 3x3
            # pixels which seems pretty small!

            # Do the convolution and save it as a new .fits file
            # interpreted_result = interpolate_replace_nans(image_data,
            #                                               kernel_stack[i])
            # conv_result = convolve_fft(interpreted_result, kernel_stack[i])

            if(kernel_stack[i].shape[0] > image_data.shape[0]):
                conv_result = convolve_fft(image_data, kernel_stack[i])
            else:
                # this was a workaround, not quite sure if this makes sense
                conv_result = convolve(image_data, kernel_stack[i])
                header['FWHM'] = (fwhm_input, "FWHM value used for Gaussian " +
                                  "convolution, in pixels")
            hdu = fits.PrimaryHDU(conv_result, header)
            hdu.writeto(convolved_filename, overwrite=True)
            image_stack[i].header = header
            image_stack[i].data = conv_result
    return image_stack


def resample_kernel(kernel_file, img_file):
    """
    Resamples the kernel to the same pixel scale as the image.

    Parameters
    ----------
    kernel_file: string
        A string containing the name of the kernel file
        that needs to be resampled

    img_file: string
        A string containing the name of the image file
        that will be convolved with this kernel

    """

    kernel = fits.open(kernel_file)[0]
    ke_pixsc = get_pixel_scale(kernel.header)

    img = fits.open(img_file)[0]
    im_pixsc = get_pixel_scale(img.header)

    lngref_input, latref_input, rotation_pa = get_ref_wcs(kernel_file)
    size_height, size_width = kernel.data.shape

    width_input = u.arcsec.to(u.deg) * ke_pixsc * size_width
    height_input = u.arcsec.to(u.deg) * ke_pixsc * size_height

    resampled_kernel = kernel_file.strip('.fits') + '_resampled.fits'

    montage.commands.mHdr(str(lngref_input) + ' ' + str(latref_input),
                          width_input, 'grid_final_resample_header',
                          system='eq', equinox=2000.0, height=height_input,
                          pix_size=im_pixsc, rotation=rotation_pa)

    artificial_header = image_directory + 'temporary_hdr.hdr'
    merge_headers('grid_final_resample_header', kernel.header,
                  artificial_header)
    montage.wrappers.reproject(kernel_file, resampled_kernel,
                               header=artificial_header, exact_size=True)

    resampled_kernel_data = fits.open(resampled_kernel)[0].data
    os.unlink(artificial_header)
    os.unlink('grid_final_resample_header')
    os.unlink(resampled_kernel)
    return resampled_kernel_data


def resample_images(image_stack, logfile_name):
    """
    Resamples all of the images to a common pixel grid.

    Parameters
    ----------
    image_stack: HDU list
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/resampled/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # figure out the geometry of the resampled images
    width_input = ang_size / im_pixsc
    height_input = width_input

    # get WCS info for the reference image
    lngref_input, latref_input, rotation_pa = get_ref_wcs(main_reference_image)

    # make the header for the resampled images (same for all)
    montage.commands.mHdr(str(lngref_input) + ' ' + str(latref_input),
                          width_input, 'grid_final_resample_header',
                          system='eq', equinox=2000.0, height=height_input,
                          pix_size=im_pixsc, rotation=rotation_pa)

    # temporary directory to store the file from image_stack
    # so that reproject function works
    tmp_directory = image_directory + "/temp/"
    if not os.path.exists(tmp_directory):
        os.mkdir(tmp_directory)

    for i in range(1, len(image_stack)):
        original_filename = os.path.basename(image_stack[i].header['FILENAME'])
        artificial_header = (new_directory + original_filename +
                             "_artheader")
        resampled_filename = (new_directory + original_filename +
                              "_resampled.fits")

        # temporary file created from image_stack
        tmp_filename = (tmp_directory + original_filename)

        hdulist = fits.HDUList()
        hdu_header = image_stack[i].header
        hdu_data = image_stack[i].data
        hdulist.append(fits.PrimaryHDU(header=hdu_header, data=hdu_data))
        hdulist.writeto(tmp_filename, overwrite=True, output_verify='ignore')

        # generate header for regridded image
        merge_headers('grid_final_resample_header',
                      image_stack[i].header, artificial_header)
        # do the regrid
        montage.wrappers.reproject(tmp_filename, resampled_filename,
                                   header=artificial_header)
        # delete the header file
        os.unlink(artificial_header)
        # remove the tmeporary file created
        os.unlink(tmp_filename)
        image_stack[i].header = fits.open(resampled_filename)[0].header
        image_stack[i].data = fits.open(resampled_filename)[0].data

    os.unlink('grid_final_resample_header')
    # remove the tmeporary directory created
    os.rmdir(tmp_directory)
    image_stack = create_data_cube(image_stack, logfile_name)
    return image_stack


def create_data_cube(image_stack, logfile_name):
    """
    Creates a data cube from the provided images.

    Parameters
    ----------
    image_stack: HDU list
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/datacube/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # put the image data into a list, not sure this is
    # quite the right way to do it
    resampled_images = []
    for i in range(1, len(image_stack)):
        hdulist = image_stack[i]
        image = hdulist.data
        resampled_images.append(image)
        if i == 1:     # grab the WCS info from the first input image
            new_wcs = wcs.WCS(hdulist.header)

    # make a new header with the WCS info
    prihdr = new_wcs.to_header()
    # put some other information in the header

    # TODO: add version
    prihdr['CREATOR'] = ('IMAGECUBE', 'Software used to create this file')
    prihdr['DATE'] = (datetime.now().strftime('%Y-%m-%d'),
                      'File creation date')
    prihdr['LOGFILE'] = (logfile_name, 'imagecube log file')
    if do_conversion:
        prihdr['BUNIT'] = ('Jy/pixel', 'Units of image data')

    # now use the header and data to create a new fits file
    prihdu = fits.PrimaryHDU(header=prihdr, data=resampled_images)
    hdulist = fits.HDUList(prihdu)

    # TODO : check why the next 2 lines are not working

    # hdulist.add_datasum(when='Computed by imagecube')
    # hdulist.add_checksum(when='Computed by imagecube',override_datasum=True)

    hdulist.writeto(new_directory + '/' + 'datacube.fits', overwrite=True)

    # if the 2D structure of the file is also to be created
    if(make_2D):
        # make a new header with the WCS info
        prihdr = new_wcs.to_header()
        # put some other information in the header

        # TODO: add version
        prihdr['CREATOR'] = ('IMAGECUBE', 'Software used to create this file')
        prihdr['DATE'] = (datetime.now().strftime('%Y-%m-%d'),
                          'File creation date')
        prihdr['LOGFILE'] = (logfile_name, 'imagecube log file')
        if do_conversion:
            prihdr['BUNIT'] = ('Jy/pixel', 'Units of image data')

        # now use this header to create a new fits file
        # put the image data into a list, not sure this is
        # quite the right way to do it
        prihdu = fits.PrimaryHDU(header=prihdr)
        cube_hdulist = fits.HDUList([prihdu])

        for i in range(1, len(image_stack)):
            hdulist = image_stack[i]
            cube_hdulist.append(hdulist)

        cube_hdulist.writeto(new_directory + '/' + 'datacube_2d.fits',
                             overwrite=True, output_verify='ignore')

    return image_stack


def output_seds(image_stack):
    """
    Makes the SEDs.

    Parameters
    ----------
    image_stack: HDU list
        A structure containing headers and image data for all FITS input
        images.

    """
    # make new directory for output, if needed
    new_directory = image_directory + "/seds/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    all_image_data = []
    wavelengths = []

    num_wavelengths = len(image_stack)
    # print("In seds")
    # print("num_wavelenghts : ", num_wavelengths)
    for i in range(1, num_wavelengths):
        wavelength = image_stack[i].header['WAVELNTH']
        wavelengths.append(wavelength)

        # Load the data for each image and append it to a master list of
        # all image data.
        # NOTETOSELF: change to use nddata structure?
        hdulist = image_stack[i]
        image_data = hdulist.data
        all_image_data.append(image_data)
    print(len(all_image_data))
    sed_data = []
    for i in range(0, num_wavelengths - 1):
        print(i, "on")
        for j in range(len(all_image_data[i])):
            for k in range(len(all_image_data[i][j])):
                sed_data.append((int(j), int(k), wavelengths[i],
                                all_image_data[i][j][k]))
        print(i, "done")

    # write the SED data to a test file
    # NOTETOSELF: make this optional?
    data = np.copy(sorted(sed_data))
    np.savetxt('test.out', data, fmt='%f,%f,%f,%f',
               header='x, y, wavelength (um), flux units (Jy/pixel)')
    num_seds = int(len(data) / num_wavelengths)

    with console.ProgressBarOrSpinner(num_seds, "Creating SEDs") as bar:
        for i in range(0, num_seds):

            # change to the desired fonts
            rc('font', family='Times New Roman')
            rc('text', usetex=True)
            # grab the data from the cube
            wavelength_values = data[:, 2][i * num_wavelengths:(i + 1) *
                                           num_wavelengths]
            flux_values = data[:, 3][i * num_wavelengths:(i + 1) *
                                     num_wavelengths]
            # NOTETOSELF: change from 0-index to 1-index

            # pixel pos
            x_values = data[:, 0][i * num_wavelengths:(i + 1) *
                                  num_wavelengths]
            y_values = data[:, 1][i * num_wavelengths:(i + 1) *
                                  num_wavelengths]
            fig, ax = plt.subplots()
            ax.scatter(wavelength_values, flux_values)
            # axes specific
            ax.set_xlabel(r'Wavelength ($\mu$m)')
            ax.set_ylabel(r'Flux density (Jy/pixel)')
            rc('axes', labelsize=14, linewidth=2, labelcolor='black')
            ax.set_xscale('log')
            ax.set_yscale('log')
            # NOTETOSELF: doesn't quite seem to work
            ax.set_xlim(min(wavelength_values), max(wavelength_values))
            ax.set_ylim(min(flux_values), max(flux_values))
            fig.savefig(new_directory + '/' + str(int(x_values[0])) + '_' +
                        str(int(y_values[0])) + '_sed.eps')
            bar.update(i)
    return


def cleanup_output_files():
    """
    Removes files that have been generated by previous executions of the
    script.
    """

    for d in ('converted', 'registered', 'convolved', 'resampled', 'seds'):
        subdir = image_directory + '/' + d
        if (os.path.isdir(subdir)):
            log.info("Removing " + subdir)
            shutil.rmtree(subdir)
    return


# if __name__ == '__main__':
def main(args=None):
    global ang_size
    global image_directory
    global main_reference_image
    global fwhm_input
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global kernel_directory
    global im_pixsc  # change variable name
    global rot_angle
    global make_2D
    ang_size = 160 #size of smallest image (arcsec)
    image_directory = 'images'
    main_reference_image = 'images/UKIRT_Kband_full.fits'
    fwhm_input = 1.4
    do_conversion = False
    do_registration = True
    do_convolution = True
    do_resampling = True
    do_seds = False
    do_cleanup = False
    kernel_directory = ''
    im_pixsc = 1.4 #largest pixel size
    rot_angle = 0 #(SAM) ADDED USER INPUT ROT ANGLE

    make_2D = True

    # note start time for log
    start_time = datetime.now()

    # parse arguments
    if args is not None:
        arglist = args.split(' ')
    else:
        arglist = sys.argv[1:]
    parse_status = parse_command_line(arglist)
    if parse_status > 0:
        if __name__ == '__main__':
            sys.exit()
        else:
            return

    if (do_cleanup):  # cleanup and exit
        cleanup_output_files()
        if __name__ == '__main__':
            sys.exit()
        else:
            return

    # NOTE_FROM_RK : A lot of these contants seem redundant and unused, need to
    #                figure out exactly which ones are used and remove the rest

    # Lists to store information
    global image_data
    global converted_data
    global registered_data
    global convolved_data
    global resampled_data
    global headers
    global filenames

    converted_data = []
    registered_data = []
    convolved_data = []
    resampled_data = []
    filenames = []
    image_data = []
    headers = []

    # append all the images before creating the stack
    hdus = []

    # First HDU in the stack, just to store some information about the stack
    hdr = fits.Header()
    hdr['COMMENT'] = "Image stack created to form the data cube"

    # this is just to allow for a later sort on the HDU list,
    # can be changed later if needed
    hdr['WAVELNTH'] = 0
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdus.append(primary_hdu)

    # if not just cleaning up, make a log file which records input parameters
    logfile_name = ("imagecube_" + start_time.strftime('%Y-%m-%d_%H%M%S') +
                    '.log')
    with log.log_to_file(logfile_name, filter_origin='imagecube.imagecube'):
        log.info('imagecube started at %s' %
                 start_time.strftime('%Y-%m-%d_%H%M%S'))
        log.info('imagecube called with arguments %s' % arglist)

        # Grab all of the .fits and .fit files in the specified directory
        all_files = glob.glob(image_directory + "/*.fit*")
        # no use doing anything if there aren't any files!
        if len(all_files) == 0:
            warnings.warn('No fits files found in directory %s' %
                          image_directory, AstropyUserWarning)
            if __name__ == '__main__':
                sys.exit()
            else:
                return

        # get images
        for (i, fitsfile) in enumerate(all_files):
            hdulist = fits.open(fitsfile)
            img_extens = find_image_planes(hdulist)
            # NOTETOSELF: right now we are just using the *first* image
            #             extension in a file which is not what we want
            #             to do, ultimately.
            header = hdulist[img_extens[0]].header
            image = hdulist[img_extens[0]].data
            # Strip the .fit or .fits extension from the filename so we
            # can append things to it later on
            filename = os.path.splitext(hdulist.filename())[0]
            hdulist.close()
            # check to see if image has reasonable scale & orientation
            # NOTETOSELF: should this really be here? It's not relevant for
            #             just flux conversion. Want separate loop over image
            #             planes, after finishing file loop
            pixelscale = get_pixel_scale(header)
            fov = pixelscale * float(header['NAXIS1'])
            log.info("Checking %s: is pixel scale (%.2f\")"
                     " < ang_size (%.2f\") < FOV (%.2f\") ?" % (fitsfile, pixelscale, ang_size, fov))
            if (pixelscale < ang_size < fov):
                try:
                    # there seems to be a different name for wavelength
                    # in some images, look into it
                    wavelength = header['WAVELNTH']
                    # add the unit if it's not already there
                    header['WAVELNTH'] = (wavelength, 'micron')
                    header['FILENAME'] = fitsfile
                    a = fits.ImageHDU(header=header, data=image)
                    hdus.append(a)
                except KeyError:
                    warnings.warn('Image %s has no WAVELNTH keyword, will not be used' % filename, AstropyUserWarning)
            else:
                warnings.warn("Image %s does not meet the above criteria." %
                              filename, AstropyUserWarning)
            # end of loop over files

        # Sort the lists by their WAVELNTH value

        hdus.sort(key=lambda x: x.header['WAVELNTH'])

        # this is the image stack, the data structure stores the images
        # in the following format :

        # Primary HDU : the first HDU contains some information on the
        #               stack created
        # Image HDU : the next 'n' image HDUs contain the headers and the data
        #             of the image files that need to be processed by IMAGECUBE

        image_stack = fits.HDUList(hdus)

        # At this step, create a kernel stack as well.
        # It should consist of the 5 kernels that need to be used to convolve.
        # Generate the kernel filename by picking up the instruments for each
        # image and the wavelength Further, before convolving each image from
        # this kernel_stack with images from the image_stack
        # Resample them so that the pixel scale match  -- DOUBT
        # Pixel scale of kernel should match with that of the image pixel scale
        kernels = []
        kernels.append([])

        # this is the url from where the kernels will be downloaded
        url0 = ("https://www.astro.princeton.edu/~ganiano/Kernels/Ker_2012/" +
                "Kernels_fits_Files/Low_Resolution/Kernel_LoRes_")

        # all the images will be transformed to the
        # PSF of the largest wavelength
        to_hdu = image_stack[-1]
        to_instr = str(to_hdu.header['INSTRUME'])
        to_wavelnth = to_hdu.header['WAVELNTH']

        # small hack since MIPS channels sometimes have wavelengths of
        # different levels of precision
        if(to_instr == "MIPS"):
            to_wavelnth = math.ceil(to_wavelnth)

        # For every image in our stack, we first look if there's a
        # corresponding kernel file in the dataset provided. If we
        # dont find one, we look for one on the URL mentioned and
        # generated using the instrument name and wavlenegth. If
        # the website does not seem to have the corresponding kernels,
        # we generate a Gaussian kernel using the
        # FWHM input and the corresponding pixel_scale

        for i in range(1, len(image_stack)):
            original_filename = os.path.basename(image_stack[i].header
                                                 ['FILENAME'])
            original_directory = os.path.dirname(image_stack[i].header
                                                 ['FILENAME'])

            kernel_filename = (original_directory + "/" + kernel_directory +
                               "/" + original_filename + "_kernel.fits")

            log.info("Looking for " + kernel_filename)

            if os.path.exists(kernel_filename):
                log.info("Found a kernel; will convolve with it shortly.")
                # reading the kernel
                kernel_hdulist = fits.open(kernel_filename)
                kernel_image = kernel_hdulist[0].data
                kernel_hdulist.close()
                kernels.append(kernel_image)

            else:
                fr_instr = str(image_stack[i].header['INSTRUME'])
                fr_wavelnth = image_stack[i].header['WAVELNTH']

                if(fr_instr == 'MIPS'):
                    fr_wavelnth = math.ceil(fr_wavelnth)

                # This is the URL generated, from where we will donwload files.

                url = (url0 + str(fr_instr) + "_" + str(fr_wavelnth) + "_to_" +
                       str(to_instr) + "_" + str(to_wavelnth) + ".fits.gz")

                filename = url.split("/")[-1]

                # TODO : Look for these files if they're already downloaded so
                # that these downloads do not need to
                # happen multiple times if the same kernel files are required.
                # Ideally, make a kernels folder to handle this
                with open(filename, "wb") as f:
                    r = requests.get(url)
                    if not r.status_code == 404:
                        f.write(r.content)
                        with gzip.open(filename, 'rb') as f_in:
                            with open(filename.split('.gz')[0], 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        log.info("File unzipped : ", filename.split('.gz')[0])

                        # resampling of the kernel, so that the file can be
                        # used for convolution
                        resampled_kernel = resample_kernel(filename.split('.gz'
                                                                          )[0],
                                                           image_stack[i]
                                                           .header['FILENAME'])
                        kernels.append(resampled_kernel)

                    else:
                        log.info("This file doesn't seem to exist on the " +
                                 "website : ",
                                 filename)
                        native_pixelscale = get_pixel_scale(image_stack[i].
                                                            header)
                        sigma_input = (fwhm_input /
                                       (2 * math.sqrt(2 * math.log(2)) *
                                        native_pixelscale))
                        kernels.append(Gaussian2DKernel(sigma_input).array)

        kernel_stack = kernels

        if (do_conversion):
            image_stack = convert_images(image_stack)

        if (do_registration):
            image_stack = register_images(image_stack)

        if (do_convolution):
            image_stack = convolve_images(image_stack, kernel_stack)

        if (do_resampling):
            image_stack = resample_images(image_stack, logfile_name)

        if (do_seds):
            output_seds(image_stack)

        # all done!
        log.info('All tasks completed.')
        if __name__ == '__main__':
            sys.exit()
        else:
            return

main()


