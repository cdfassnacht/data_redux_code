"""
spec_simple.py - A library of functions to do various basic CCD spectroscopy
  processing operations

Functions:
   xxxx  - descriptions here
"""

from math import sqrt,pi
import pyfits
import numpy as n
import numpy as np
import scipy as sp
from scipy import optimize,interpolate,ndimage
import numpy
import matplotlib.pyplot as plt
import glob
import ccdredux as c
#import nirspec

#-----------------------------------------------------------------------

def clear_all(nfig=10):
   """
   Clears nfig figures
   """

   for i in range(nfig):
      plt.figure(i+1)
      plt.clf()

#-----------------------------------------------------------------------

def find_blank_columns(data,comp_axis=0,output_dims=1,findblank=False):
   """
   Takes 2-dimensional data and outputs indices of columns not entirely
   composed of zeros. If output_dims is 1, only the indices of the columns
   are give. If output_dims = 2, the indices for every point in any of these
   columns are given. By default, it is actually non-blank columns that are
   found. Setting findblank to True switches this.
   """
   if data.ndim != 2: sys.exit("find_blank_columns takes only 2-dimensional data")
   if output_dims==1:
      fbc_tmp = np.zeros(n.shape(data)[1-comp_axis])
      if comp_axis == 0:
         gprelim = np.where(data[int(n.shape(data)[comp_axis]/2),:] == 0)[0]
         for ifbc in range(0,len(gprelim)):
            if len(data[data[:,gprelim[ifbc]] != 0]) == 0: fbc_tmp[gprelim[ifbc]] = 1
      else:
         gprelim = np.where(data[:,int(n.shape(data)[comp_axis]/2)] == 0)[0]
         for ifbc in range(0,len(gprelim)):
            if len(data[data[gprelim[ifbc],:] != 0]) == 0: fbc_tmp[gprelim[ifbc]] = 1
      if findblank: fbc_tmp = 1-fbc_tmp
      gfbc = np.where(fbc_tmp == 0)[0]
   elif output_dims==2:
      fbc_tmp = n.zeros(n.shape(data))
      if comp_axis == 0:
         gprelim = np.where(data[int(n.shape(data)[comp_axis]/2),:] == 0)[0]
         for ifbc in range(0,len(gprelim)):
            if len(data[data[:,gprelim[ifbc]] != 0]) == 0: fbc_tmp[:,gprelim[ifbc]] = 1
      else:
         gprelim = np.where(data[:,int(n.shape(data)[comp_axis]/2)] == 0)[0]
         for ifbc in range(0,len(gprelim)):
            if len(data[data[gprelim[ifbc],:] != 0]) == 0: fbc_tmp[gprelim[ifbc],:] = 1
      if findblank: fbc_tmp = 1-fbc_tmp
      gfbc = np.where(fbc_tmp == 0)
   else:
      sys.exit("output_dims parameter for find_blank_columns must be either 1 or 2. Value was: " + str(output_dims))
   return gfbc


#-----------------------------------------------------------------------

def load_raw_spectrum(filename):
   """
   Reads in a raw 2D spectrum from a fits file
   """
   data = pyfits.open(filename)[0].data
   return data

#-----------------------------------------------------------------------

def read_spectrum(filename, informat='text', varspec=True, verbose=True, line=1):
   """
   Reads in an extracted 1D spectrum and possibly the associated variance
   spectrum.  There are two possible input file formats:
      mwa:  A multi-extension fits file with wavelength info in the fits header
            Extension 1 is the extracted spectrum (flux)
            Extension 3 is the variance spectrum
      text: An ascii text file with information in columns:
            Column 1 is the wavelength
            Column 2 is the extracted spectrum
            Column 3 (optional) is the variance spectrum

   Inputs:
      filename - input file name
      informat - format of input file ("mwa" or "text")
      varspec  - if informat is text, then this sets whether to read in a 
                 variance spectrum.  Default: varspec = True
   """

   if verbose:
      print ""
      print "Reading spectrum from %s" % filename

   if informat=="mwa":
      hdulist = pyfits.open(filename)
      flux = hdulist[1].data.copy()
      var  = hdulist[3].data.copy()
      varspec = True
      hdr1 = hdulist[1].header
      wavelength = n.arange(flux.size) - hdr['crpix1']
      wavelength = hdr1['crval1'] + wavelength*hdr1['cd1_1']
      del hdulist
   else:
      spec = numpy.loadtxt(filename)
      wavelength = spec[:,0]
      if line == 1:
         flux       = spec[:,1]
         if varspec:
            var     = spec[:,2]
      else:
         if varspec:
            flux = spec[:,2*line-1]
            var = spec[:,2*line]
         else:
            flux = spec[:,line]
      del spec

   if verbose:
      print " Spectrum Start: %8.2f" % wavelength[0]
      print " Spectrum End:   %8.2f" % wavelength[-1]
      print " Dispersion (1st pixel): %6.2f" % (wavelength[1]-wavelength[0])
      print " Dispersion (average):   %6.2f" % \
          ((wavelength[-1]-wavelength[0])/(wavelength.size-1))
      print ""

   if varspec:
      return wavelength, flux, var
   else:
      return wavelength, flux, None

#-----------------------------------------------------------------------

def save_spectrum(filename,x,flux,var=None):
   """
   Saves a spectrum as a text file
   """
   if len(np.shape(flux)) == 1:
      if var is not None:
         outdata = numpy.zeros((x.shape[0],3))
         outdata[:,2] = var
         fmtstring = '%7.2f %9.3f %10.4f'
      else:
         outdata = numpy.zeros((x.shape[0],2))
         fmtstring = '%7.2f %9.3f'
      outdata[:,1] = flux
      outdata[:,0] = x
   else:
      if var is not None:
         outdata = numpy.zeros((x.shape[1],1+2*np.shape(flux)[0]))
         fmtstring = '%7.2f %9.3f %10.4f'
         outdata[:,2] = var[0]
         for iss in range(1,np.shape(flux)[0]):
            fmtstring = fmtstring + ' %9.3f %10.4f'
            outdata[:,2*(iss+1)] = var[iss]
            outdata[:,2*iss+1] = flux[iss]
      else:
         outdata = numpy.zeros((x.shape[1],1+np.shape(flux)[0]))
         fmtstring = '%7.2f %9.3f'
         for iss in range(1,np.shape(flux)[0]):
            fmtstring = fmtstring + ' %9.3f'
            outdata[:,iss+1] = flux[iss]
      outdata[:,1] = flux[0]
      outdata[:,0] = x[0]
   print ""
   print "Saving spectrum to file %s" % filename
   numpy.savetxt(filename,outdata,fmt=fmtstring)
   del outdata

#-----------------------------------------------------------------------

def plot_spectrum_array(x, flux, var=None, xlabel="Wavelength (Angstroms)",
                        ylabel="Relative Flux", title='Extracted Spectrum', 
                        docolor=True, rmsoffset=0, rmsls=None, fontsize=12):

   """
   Given two input arrays, plot a spectrum.
   """

   if docolor:
      speccolor = 'b'
      rmscolor  = 'r'
   else:
      speccolor = 'k'
      rmscolor  = 'k'
   plt.axhline(color='k')
   plt.plot(x,flux,speccolor,linestyle='steps',label='Flux')
   plt.tick_params(labelsize=fontsize)
   plt.xlabel(xlabel,fontsize=fontsize)
   if var is not None:
      rms = numpy.sqrt(var)+rmsoffset
      if rmsls is None:
         if docolor:
            rlinestyle = 'steps'
         else:
            rlinestyle = 'steps:'
      else:
         rlinestyle = 'steps%s' % rmsls
      if docolor:
         plt.plot(x,rms,rmscolor,linestyle=rlinestyle,label='RMS')
      else:
         plt.plot(x,rms,rmscolor,linestyle=rlinestyle,label='RMS',lw=2)
   plt.ylabel(ylabel,fontsize=fontsize)
   if(title):
      plt.title(title)
   if(x[0] > x[-1]):
      plt.xlim([x[-1],x[0]])
   else:
      plt.xlim([x[0],x[-1]])

#-----------------------------------------------------------------------

def plot_spectrum(filename, varspec=True, informat="text", 
                  xlabel="Wavelength (Angstroms)", ylabel="Relative Flux",
                  title="Extracted Spectrum",
                  fontsize=12, docolor=True, rmsoffset=0, rmsls=None,
                  add_atm_trans=False, atmscale=1.05, atmfwhm=15.,
                  atmoffset=0., atmls='-',verbose=True,line=1,output=False,clear=False):
   """
   Given an input file with spectroscopy information, plot a spectrum.  
   The input file can have one of two formats: text (the default) or mwa
   For the text format, the input file is an ascii text file, with 2 or 3 columns
      1. Wavelength (or pixel position in the dispersion direction)
      2. Flux (counts, etc.)
      3. Variance spectrum (OPTIONAL)
   The mwa format is created by the make_spec function in the spectools
    library (probably found in the mostools package).
    The format is a multi-extension FITS file:
      Extension 1: the science spectrum
      Extension 2: a smoothed version of the science spectrum
      Extension 3: the variance spectrum.
    NOTE: For this format, the wavelength information is contained in the
      FITS header cards, and is not stored as a separate array.
   """

   """ Read in the spectrum, using the appropriate input format """
   if clear: plt.clf()
   wavelength,flux,var = read_spectrum(filename, informat, varspec, verbose, line=line)

   """ Plot the spectrum """
   if varspec:
      plot_spectrum_array(wavelength,flux,var=var,xlabel=xlabel,ylabel=ylabel,
                          title=title,docolor=docolor,rmsoffset=rmsoffset,
                          rmsls=rmsls,fontsize=fontsize)
   else:
      plot_spectrum_array(wavelength,flux,xlabel=xlabel,ylabel=ylabel,
                          title=title,docolor=docolor,rmsoffset=rmsoffset,
                          fontsize=fontsize)

   """ Plot the atmospheric transmission if requested """
   if add_atm_trans:
      plot_atm_trans(wavelength, atmfwhm, flux, scale=atmscale, 
                     offset=atmoffset, linestyle=atmls)

   del wavelength
   del flux
   if(varspec):
      del var
   if output:
      return wavelength,flux,var

#-----------------------------------------------------------------------

def make_gauss_plus_bkgd(x,mu,sigma,amp,bkgd):
   """
   Creates a gaussian plus background model given input x and parameters.
   The parameter values are:
    mu
    sigma
    amplitude
    background level
   """

   """ Calculate y_mod using current parameter values """
   if ((np.shape(bkgd) == ()) & (np.shape(amp) == ())):
      ymod = bkgd + amp * numpy.exp(-0.5 * ((x - mu)/sigma)**2)
   elif np.shape(bkgd) == ():
      x,amp,mu,sigma = x*np.transpose([np.ones(len(mu))]),np.transpose([amp]),np.transpose([mu]),np.transpose([sigma])
      ymod = bkgd + np.sum(amp * numpy.exp(-0.5 * ((x - mu)/sigma)**2),axis=0)
   else:
      x,amp,mu,sigma = x*np.transpose([np.ones(len(mu))]),np.transpose([amp]),np.transpose([mu]),np.transpose([sigma])
      ymod = bkgd[0] + np.sum(amp * numpy.exp(-0.5 * ((x - mu)/sigma)**2),axis=0)

   return ymod

#-----------------------------------------------------------------------

def fit_gauss_plus_bkgd(p,x,y):
   """
   Compares the data to the model.  The model is a gaussian plus a 
    constant background.
   The parameter values are:
    p[0] = background level
    p[1] = mu
    p[2] = sigma
    p[3] = amplitude
   This function now supports multiple gaussians. p[1-3] must be arrays
   that contain the respective values of mu, sigma, and amplitude for 
   the different gaussians. 
   """

   """ Unpack p """
   bkgd = p[0]
   mu   = p[1]
   sig  = p[2]
   amp  = p[3]
   if len(p) > 4:
      nps = (len(p)-1)/3
      for inpsf in range(1,nps):
         mu,amp,sig = np.append(mu,p[3*inpsf+1]),np.append(amp,p[3*inpsf+2]),np.append(sig,p[3*inpsf+3])

   """
   Compute the difference between model and real values
   """

   ymod = make_gauss_plus_bkgd(x,mu,sig,amp,bkgd)
   diff = y - ymod

   return diff

#-----------------------------------------------------------------------

def fit_gpb_fixmusig(p,x,y,mu,sigma):
   """
   Compares the data to the model.  The model is a gaussian plus a 
    constant background.  In the fit, mu and sigma are held fixed.
   The parameter values are:
    p[0] = background level
    p[1] = amplitude
   """

   """ Unpack p """
   bkgd = p[0]
   amp  = p[1]
   if len(p) > 2: amp = p[1:]
   """
   Compute the difference between model and real values
   """
   if np.shape(amp) != (): sigma,mu = np.ones(len(bkgd))*sigma,np.ones(len(bkgd))*mu
   ymod = make_gauss_plus_bkgd(x,mu,sigma,amp,bkgd)
   diff = y - ymod

   return diff

#-----------------------------------------------------------------------

#-----------------------------------------------------------------------

def fit_gpb_fixmu(p,x,y,mu):
   """
   Compares the data to the model.  The model is a gaussian plus a 
    constant background.  In the fit, mu is held fixed.
   The parameter values are:
    p[0] = background level
    p[1] = amplitude
    p[2] = sigma
   """

   """ Unpack p """
   bkgd = p[0]
   amp  = p[1]
   sigma = p[2]
   if len(p) > 3:
      nps = (len(p)-1)/2
      for inpsf in range(1,nps):
         amp,sigma = np.append(amp,p[2*inpsf+1]),np.append(sigma,p[2*inpsf+2])

   """
   Compute the difference between model and real values
   """
   if np.shape(amp) != (): mu = np.ones(len(bkgd))*mu
   ymod = make_gauss_plus_bkgd(x,mu,sigma,amp,bkgd)
   diff = y - ymod

   return diff

#-----------------------------------------------------------------------

def plot_spatial_profile(infile, dispaxis="x"):
   """
   Given an input fits file with (by assumption) a 2d spectrum, this
   function will compress the spectrum along the dispersion direction
   to get an average spatial profile.

   Inputs:
      infile   - input file containing the 2d spectrum
      dispaxis - axis corresponding to the dispersion direction (i.e.,
                 the wavelength axis)
   """

   # Read the data
   data = load_raw_spectrum(infile)

   # Set the dispersion axis direction
   if dispaxis == "y":
      specaxis = 0
      spatlabel = "x"
   else:
      specaxis = 1
      spatlabel = "y"
   #print "specaxis = %d" % specaxis

   """ Compress the data along the dispersion axis and find the max value """
   if data.ndim < 2:
      print ""
      print "ERROR: plot_spatial_profile needs a 2 dimensional data set"
      del data
      return
   else:
      cdat = numpy.median(data,axis=specaxis)
   x = numpy.arange(1,cdat.shape[0]+1)

   """ Plot the spatial profile """
   plt.plot(x,cdat)
   plt.xlabel("Pixel in the %s direction" % spatlabel)
   plt.ylabel("Median counts")
   plt.title("Spatial profile for %s" % infile)

#-----------------------------------------------------------------------

def find_peak(data,dispaxis="x",mu0=None,sig0=None,fixmu=False,fixsig=False,
   showplot=True,do_subplot=False,verbose=True,apmin=-4.,apmax=4.,noblankcolumns=True,findmultiplepeaks=False,nofit=False):
   """
    Compresses a 2d spectrum along the dispersion axis so that
     the trace of the spectrum can be automatically located by fitting
     a gaussian + background to the spatial direction.  The function
     returns the parameters of the best-fit gaussian.
    The default dispersion axis is along the x direction.  To change this
     set the optional parameter dispaxis to "y"
    If noblankcolumns is True, any column entirely composed of zeros is not
     used for analysis
   """

   # Set the dispersion axis direction
   if dispaxis == "y":
      specaxis = 0
   else:
      specaxis = 1
   #print "specaxis = %d" % specaxis

   """ Compress the data along the dispersion axis and find the max value """
   if data.ndim < 2:
      gfbc_all = np.where(data > np.min(data)-2.)[0]
      cdat = data[gfbc_all]
   else:
      if noblankcolumns:
         gfbc_rows = find_blank_columns(data)
         gfbc_all = find_blank_columns(data,output_dims=2)
      else:
         gfbc_rows = np.arange(np.shape(data)[specaxis])
         gfbc_all = np.where(data > np.min(data)-2.)
      if specaxis == 1:
         cdat = numpy.median(data[:,gfbc_rows],axis=specaxis)
      else:
         cdat = numpy.median(data[gfbc_rows,:],axis=specaxis)
      cdat.shape
   x = numpy.arange(1,cdat.shape[0]+1)

   # Set initial guesses

   if fixmu:
      if mu0 is None:
         print ""
         print "ERROR: find_peak.  mu is fixed, but no value for mu0 given"
         return
      fixmunote = "**"
   else:
      if mu0 is None:
         i = cdat.argsort()
         mu0    = 1.0 * i[i.shape[0]-1]
      fixmunote = " "
   if fixsig:
      if sig0 is None:
         print ""
         print "ERROR: find_peak.  sigma is fixed, but no value for sig0 given"
         return
      fixsignote = "**"
   else:
      if sig0 is None:
         sig0 = 3.0
         if np.shape(mu0) != (): sig0 *= np.ones(len(mu0))
      fixsignote = " "
   amp0  = cdat.max()
   if np.shape(mu0) != ():
      amp0 = np.zeros(len(mu0))
      amp0[0] = cdat.max()
      for iai in range(1,len(mu0)):
         giai = np.arange(len(cdat))
         giai = giai[(giai >= mu0[iai]+apmin-1) & (giai <= mu0[iai]+apmax)]
         amp0[iai] = cdat[giai].max()
   bkgd0 = numpy.median(data[gfbc_all],axis=None)
   if verbose and not nofit and np.shape(mu0) == ():
      print ""
      print "Initial guesses for Gaussian plus background fit"
      print "------------------------------------------------"
      print " mu         = %7.2f%s"   % (mu0,fixmunote)
      print " sigma      =   %5.2f%s" % (sig0,fixsignote)
      print " amplitude  = %f"        % amp0
      print " background = %f"        % bkgd0
      print "Parameters marked with a ** are held fixed during the fit"
      print ""

   # Fit a Gaussian plus a background to the compressed spectrum
   mf=100000
   if not nofit:
      if np.shape(bkgd0) != (): bkgd0 = bkgd0[0]
      if fixmu and fixsig:
         if np.shape(amp0) == ():
            p = [bkgd0,amp0]
         else:
            p = np.append(bkgd0,amp0)
         pt,ier = optimize.leastsq(fit_gpb_fixmusig,p,(x,cdat,mu0,sig0),maxfev=mf)
         ampout = pt[1:]
         if np.shape(amp0) == ():
            bkgdout = pt[0]
         else:
            bkgdout *= np.ones(len(ampout))
         p_out = [bkgdout,mu0,sig0,ampout]
   #p_out,ier = optimize.leastsq(fit_gpb_fixmu,p,(x,cdat,mu0),maxfev=mf)
   #p_out,ier = optimize.leastsq(fit_gpb_fixsig,p,(x,cdat,sig0),maxfev=mf)
      elif fixmu and not fixsig:
         if np.shape(amp0) == ():
            p = [bkgd0,amp0,sig0]
         else:
            p = [bkgd0]
            for ifmfs in range(0,len(amp0)):
               p = np.append(p,[amp0[ifmfs],sig0[ifmfs]])
         pt,ier = optimize.leastsq(fit_gpb_fixmu,p,(x,cdat,mu0),maxfev=mf)
         if len(pt) > 3:
            nps = (len(pt)-1)/2
            ampout,sigout = np.zeros(0),np.zeros(0)
            for inps in range(0,nps):
               ampout,sigout = np.append(ampout,pt[2*inps+1]),np.append(sigout,pt[2*inps+2])
            p_out = [pt[0]*np.ones(len(mu0)),mu0,sigout,ampout]
         else:
            p_out = [pt[0],mu0,pt[2],pt[1]]
      else:
         if np.shape(mu0) == ():
            p = [bkgd0,mu0,sig0,amp0]
            p_out,ier = optimize.leastsq(fit_gauss_plus_bkgd,p,(x,cdat),maxfev=mf)
         else:
            p = [bkgd0]
            for ifmfs in range(0,len(amp0)):
               p = np.append(p,[mu0[ifmfs],amp0[ifmfs],sig0[ifmfs]])
               p_out,ier = optimize.leastsq(fit_gauss_plus_bkgd,p,(x,cdat),maxfev=mf)
               nps = (len(p_out)-1)/3
            ampout,sigout,muout = np.zeros(0),np.zeros(0),np.zeros(0)
            for inps in range(0,nps):
               muout,ampout,sigout = np.append(muout,p_out[3*inps+1]),np.append(ampout,p_out[3*inps+2]),np.append(sigout,p_out[3*inps+2])
            p_out = [p_out[0]*np.ones(len(mu0)),mu0,sigout,ampout]

   # Give results
   if verbose and not nofit and np.shape(p_out[1]) == ():
      print "Fitted values for Gaussian plus background fit"
      print "----------------------------------------------"
      print " mu         = %7.2f%s"   % (p_out[1],fixmunote)
      print " sigma      =   %5.2f%s" % (p_out[2],fixsignote)
      print " amplitude  = %f"        % p_out[3]
      print " background = %f"        % p_out[0]
      print "Parameters marked with a ** are held fixed during the fit"
      print ""

   # Plot the compressed spectrum
   if showplot and not nofit:
      if(do_subplot):
         plt.figure(2)
         plt.subplot(221)
      else:
         plt.figure(2)
         plt.clf()
      if np.shape(p_out[0]) == 0:
         bkgd = p_out[0]
      else:
         bkgd = np.sum(p_out[0])
      plt.plot(x,cdat,linestyle='steps')
      xmod = numpy.arange(1,cdat.shape[0]+1,0.1)
      ymod = make_gauss_plus_bkgd(xmod,p_out[1],p_out[2],p_out[3],p_out[0])
      plt.plot(xmod,ymod)
      if np.shape(p_out[1]) == ():
         plt.axvline(p_out[1]+apmin,color='k')
         plt.axvline(p_out[1]+apmax,color='k')
      else:
         for ipap in range(0,len(p_out[1])):
            plt.axvline(p_out[1][ipap]+apmin,color='k')
            plt.axvline(p_out[1][ipap]+apmax,color='k')
      plt.xlabel('Pixel number in the spatial direction')
      plt.title('Compressed Spatial Plot')
   elif showplot and nofit:
      if(do_subplot):
         plt.figure(2)
         plt.subplot(221)
      else:
         plt.figure(2)
         plt.clf()
      plt.plot(x,cdat,linestyle='steps')
      plt.xlabel('Pixel number in the spatial direction')
      plt.title('Compressed Spatial Plot')
      

   if nofit:
      return np.array([bkgd0,mu0,sig0,amp0])
   else:
      return p_out

#-----------------------------------------------------------------------

def extract_wtsum_col(spatialdat,mu,apmin,apmax,weight='gauss',sig=1.0,
                      gain=1.0,rdnoise=0.0,sky=None, weighted_var=True,var_in=None,bkgd=None,amp0=None):
   """
   Extracts the spectrum from one row/column in the wavelength direction
   via a weighted sum.  The choices for the weighting are:
       'gauss'   - a Gaussian, where mu and sigma of the Gaussian are fixed.
       'uniform' - uniform weighting across the aperture, which is centered
                   at mu

   Inputs:
     spatialdat - a one-dimensional array, corresponding to a cut in
                  the spatial direction from the 2-d spectrum
     mu         - the fixed centroid of the trace
     apmin      - the lower bound, with respect to mu, of the aperture to be
                  extracted
     apmax      - the upper bound, with respect to mu, of the aperture to be
                  extracted
     weight     - the weighting scheme to be used.  Valid choices are:
                  'gauss'   (the default value)
                  'uniform' 
     sig        - the fixed sigma of the Gaussian fit to the trace
     gain       - CCD gain - used to compute the variance spectrum
     rdnoise    - CCD readnoise  - used to compute the variance spectrum
     sky        - sky value for this wavelength (default=None).  Used only
                  if the spectrum passed to this function has already been
                  background-subtracted.
     weighted_var - If true, the variance is calculated using the same weighting
                  scheme as for spatialdat
     var_in     - If using weighted_var, an input variance of the same 
                  dimensions as spatialdat must be input here.
   """

   """ Define aperture and background regions """
   #apstart = int(mu-apsize/2.0)
   #apend   = apstart+apsize+1
   if np.shape(mu) == ():
      apstart = int(mu+apmin)
      apend   = int(mu+apmax+1.)
      apmask = numpy.zeros(spatialdat.shape,dtype=bool)
      apmask[apstart:apend] = True
      bkgdmask = numpy.logical_not(apmask)
      """ Estimate the background """
      if bkgd == None: bkgd = numpy.median(spatialdat[bkgdmask],axis=None)
   #print "Background level is %7.2f" % bkgd

      """ Make the weight array """
      y = numpy.arange(spatialdat.shape[0])
      if(weight == 'uniform'):
         gweight = numpy.zeros(y.size)
         gweight[apmask] = 1.0
      else:
         gweight = make_gauss_plus_bkgd(y,mu,sig,1.0,0.0)

         """ Do the weighted sum """
      wtsum = ((spatialdat - bkgd)*gweight)[apmask].sum() / gweight[apmask].sum()

      """ Calculate the variance """
      if weighted_var:
         var = (var_in*(gweight)**2)[apmask].sum() / (gweight[apmask].sum())**2
      elif (sky == None):
         varspec = (gain * spatialdat + rdnoise**2)/gain**2
         var = (varspec * gweight)[apmask].sum() / gweight[apmask].sum()
      else:
         varspec = (gain * (spatialdat + sky) + rdnoise**2)/gain**2
         var = (varspec * gweight)[apmask].sum() / gweight[apmask].sum()
   else:
      wtsum,var = np.zeros(len(mu)),np.zeros(len(mu))
      for iex in range(0,len(mu)):
         apstart = int(mu[iex]+apmin)
         apend   = int(mu[iex]+apmax+1.)
         apmask = numpy.zeros(spatialdat.shape,dtype=bool)
         apmask[apstart:apend] = True
         bkgdmask = numpy.logical_not(apmask)
         """ Estimate the background """
         if bkgd == None: bkgd = numpy.median(spatialdat[bkgdmask],axis=None)

         """ Make the weight array """
         y = numpy.arange(spatialdat.shape[0])
         if(weight == 'uniform'):
            gweight = numpy.zeros(y.size)
            gweight[apmask] = 1.0
         else:
            gweight = make_gauss_plus_bkgd(y,mu[iex],sig[iex],1.0,0.0)

         """ Do the weighted sum """
         wtsum[iex] = ((spatialdat - bkgd)*gweight)[apmask].sum() / gweight[apmask].sum()
         if amp0 != None:
            gamp = np.arange(len(mu))[np.arange(len(mu)) != iex]
            wtsum[iex] -= np.sum((np.transpose(np.array([amp0[gamp]]))*np.exp(-0.5*(np.arange(1,len(spatialdat)+1)[apmask]-np.transpose(np.array([mu[gamp]])))**2/np.transpose(np.array([sig]))))*gweight[apmask])/gweight[apmask].sum()
            #if gweight[apmask].sum() == 0: 
               #print 'gweight sum is zero: Point A'
               #print gweight[apmask],mu[iex],sig[iex]

         """ Calculate the variance """
         if weighted_var:
            var[iex] = (var_in*(gweight)**2)[apmask].sum() / (gweight[apmask].sum())**2
            #if amp0 != None:
               #print "Variance calculated without accounting for subtraction of multiple gaussians"
            #if gweight[apmask].sum() == 0: print 'gweight sum is zero: Point B'
         elif (sky == None):
            varspec = (gain * spatialdat + rdnoise**2)/gain**2
            var[iex] = (varspec * gweight)[apmask].sum() / gweight[apmask].sum()
         else:
            varspec = (gain * (spatialdat + sky) + rdnoise**2)/gain**2
            var[iex] = (varspec * gweight)[apmask].sum() / gweight[apmask].sum()
      
   return wtsum, var

#-----------------------------------------------------------------------

def plot_multiple_peaks(cdat,tp,theight,apmin=-4.,apmax=4.,maxpeaks=2,fig=4,clearfig=True,plot_fits=True):
   plt.figure(fig)
   if clearfig: plt.clf()
   plt.plot(np.arange(1,theight+1),cdat,linestyle='steps',color='black')
   xmod = numpy.arange(1,theight+1,0.1)
   tcolors = np.array(['red','cyan','magenta','green','blue','yellow'])
   for ipg in range(0,maxpeaks):
      ymod = make_gauss_plus_bkgd(xmod,tp[ipg][1],tp[ipg][2],tp[ipg][3],tp[0][0])
      if plot_fits: plt.plot(xmod,ymod,color=tcolors[ipg])
      plt.axvline(tp[ipg][1]+apmin,color=tcolors[ipg])
      plt.axvline(tp[ipg][1]+apmax,color=tcolors[ipg])
      if tp[ipg][3]*1.05 > 2*np.max(cdat):
         plt.text(tp[ipg][1],1.8*np.max(cdat),str(ipg+1),color=tcolors[ipg])
      elif ((tp[ipg][3]*1.05 < 2*np.min(cdat)) & (tp[ipg][3]*1.05 < -2*np.max(cdat))):
         plt.text(tp[ipg][1],np.min([1.8*np.min(cdat),-1.8*np.max(cdat)]),str(ipg+1),color=tcolors[ipg])
      else:
         plt.text(tp[ipg][1],tp[ipg][3]*1.05,str(ipg+1),color=tcolors[ipg])
   if plt.ylim()[1] > 2*np.max(cdat):
      plt.ylim(plt.ylim()[0],2*np.max(cdat))
   if ((plt.ylim()[0] < 2*np.min(cdat)) & (plt.ylim()[0] < -2*np.max(cdat))):
      plt.ylim(np.min([2*np.min(cdat),-2*np.max(cdat)]),plt.ylim()[1])
   plt.xlabel('Pixel number in the spatial direction')
   plt.title('Compressed Spatial Plot with Potential Peaks')

#-----------------------------------------------------------------------

def find_multiple_peaks(data,dispaxis="x",apmin=-4.,apmax=4.,maxpeaks=2,output_plot=None,output_plot_dir=None):
   tdata = data.copy()
   gfbc = find_blank_columns(tdata)
   if dispaxis == 'x':
      data[:,gfbc]
   tp = np.zeros((maxpeaks,4,))
   p_prelim = find_peak(tdata,dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=False,do_subplot=False,nofit=True)
   tp[0] = p_prelim
   for ifmp in range(1,maxpeaks):
      if dispaxis == 'x':
         theight = np.shape(tdata[:,gfbc])[0]
         tlength = np.shape(tdata[:,gfbc])[1]
         if ifmp == 1: x = np.arange(1,theight+1)
         gx = np.where((x < p_prelim[1]+2*apmin) | (x > p_prelim[1]+2*apmax))[0]
         if ifmp != 1: gx = np.intersect1d(gx,gxprev)
         p_prelim = find_peak(tdata[gx,:],dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=False,do_subplot=False,nofit=True)
         tp[ifmp] = p_prelim
         tp[ifmp][1] = x[gx[int(tp[ifmp][1])-1]]
         gxprev = gx.copy()
      else:
         tlength = np.shape(tdata[gfbc,:])[0]
         theight = np.shape(tdata[gfbc,:])[1]
         if ifmp == 1: x = np.arange(1,theight+1)
         gx = np.where((x < p_prelim[1]+2*apmin) | (x > p_prelim[1]+2*apmax))[0]
         if ifmp != 1: gx = np.intersect1d(gx,gxprev)
         p_prelim = find_peak(tdata[:,gxprev],dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=False,do_subplot=False,nofit=True)
         tp[ifmp] = p_prelim
         tp[ifmp][1] = x[gx[int(tp[ifmp][1])-1]]
         gxprev = gx.copy()
   tp = find_peak(tdata,dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=False,do_subplot=False,mu0=tp[:,1])
   tp[0] = np.ones(len(tp[1]))*tp[0]
   tp = np.transpose(tp)
   if dispaxis == 'x':
      cdat = numpy.median(data[:,gfbc],axis=1)
   else:
      cdat = numpy.median(data[gfbc,:],axis=0)
   plot_multiple_peaks(cdat,tp,theight,apmin=apmin,apmax=apmax,maxpeaks=maxpeaks)
   print 'Plotting %i highest peaks found\n'%maxpeaks
   tflag,fitmp,fixmu = False,False,False
   while not tflag:
      inp_fitmp = raw_input('Reduce secondary peaks? (y/n)\n')
      if ((inp_fitmp == 'y') | (inp_fitmp == 'Y')):
         tflag,fitmp = True,True
      elif ((inp_fitmp == 'n') | (inp_fitmp == 'N')):
         tflag = True
      elif inp_fitmp == 'fixmu':
         tflag,fixmu = True,True
      else:
         print 'Invalid input\n'
   fitpeaks = np.zeros(maxpeaks,dtype='bool')
   fitpeaks[0] = True
   bounds_arr = np.array([0,np.min(np.shape(data))])
   if fitmp:
      tflag = False
      while not tflag:
         inp_chp1 = raw_input("Is peak 1 okay? (y/n)\n")
         if ((inp_chp1 == 'y') | (inp_chp1 == 'Y')):
            tflag = True
         elif((inp_chp1 == 'n') | (inp_chp1 == 'N')):
            mflag = False
            while not mflag:
               inp_newp = raw_input("Current mu for peak 1 is %f. Is this acceptable? Enter 'y' or new value for mu.\n"%(tp[0][1]))
               if ((inp_newp == 'y') | (inp_newp == 'Y')):
                  mflag,tflag = True,True
               else:
                  try:
                     tp[0][1] = float(inp_newp)
                     plot_multiple_peaks(cdat,tp,theight,apmin=apmin,apmax=apmax,maxpeaks=maxpeaks)
                  except ValueError:
                     print 'Invalid input\n'
         else:
            print 'Invalid input\n'
      for iwp in range(1,maxpeaks):
         tflag = False
         while not tflag:
            inp_whichp = raw_input("Reduce peak %i? (y/n/manual) Enter 'manual' to manually set peak position\n"%(iwp+1))
            if ((inp_whichp == 'y') | (inp_whichp == 'Y')):
               tflag,fitpeaks[iwp] = True,True
            elif((inp_whichp == 'n') | (inp_whichp == 'N')):
               tflag = True
            elif ((inp_whichp == 'manual') | (inp_whichp == 'm')):
               mflag = False
               while not mflag:
                  inp_newp = raw_input("Current mu for peak %i is %f. Is this acceptable? Enter 'y' or new value for mu.\n"%(iwp+1,tp[iwp][1]))
                  if ((inp_newp == 'y') | (inp_newp == 'Y')):
                     mflag = True
                     tflag,fitpeaks[iwp] = True,True
                  else:
                     try:
                        tp[iwp][1] = float(inp_newp)
                        plot_multiple_peaks(cdat,tp,theight,apmin=apmin,apmax=apmax,maxpeaks=maxpeaks)
                     except ValueError:
                        print 'Invalid input\n'
            else:
               print 'Invalid input\n'
      num_peaks = len(fitpeaks[fitpeaks])
      mp_out = np.zeros((4,num_peaks))
      for impo in range(0,maxpeaks): 
         if fitpeaks[impo]: 
            inow = len(fitpeaks[0:impo+1][fitpeaks[0:impo+1]])
            mp_out[:,inow-1] = tp[inow-1]
      mus_tmp = mp_out[1]
      sort_mus = np.sort(mus_tmp)
      argsort_mus = np.argsort(mus_tmp)
      tbounds_arr = np.zeros(2*num_peaks)
      tbounds_arr[2*num_peaks-1] = np.min(np.shape(data))
      for il in range(0,num_peaks-1): tbounds_arr[2*il+1:2*il+3] = np.mean(sort_mus[il:il+2])
      bounds_arr = np.zeros(2*num_peaks)
      aa_mus = np.argsort(argsort_mus)
      for il in range(0,num_peaks): bounds_arr[2*il:2*il+2] = tbounds_arr[2*aa_mus[il]:2*aa_mus[il]+2]
      plot_multiple_peaks(cdat,tp,theight,apmin=apmin,apmax=apmax,maxpeaks=num_peaks)
      for il in range(0,2*num_peaks): plt.axvline(bounds_arr[il],color='k')
      
      tflag = False
      inp_aps = raw_input("Are these bounds okay? (y/n)\n")
      while not tflag:
         if ((inp_aps == 'y') | (inp_aps == 'Y')):
            tflag = True
         elif ((inp_aps == 'n') | (inp_aps == 'N')):
            for ilf in range(0,num_peaks):
               tflag2 = False
               inp_aps2 = raw_input("Are the bounds for peak %i okay? (y/n)\n"%(ilf+1))
               while not tflag2:
                  if ((inp_aps2 == 'y') | (inp_aps2 == 'Y')):
                     tflag2 = True
                  elif ((inp_aps2 == 'n') | (inp_aps2 == 'N')):
                     tflag3 = False
                     nlb = raw_input("Current bounds for peak %i are (%.1f,%.1f). Enter new lower bound:\n"%(ilf+1,bounds_arr[2*ilf],bounds_arr[2*ilf+1]))
                     nub = raw_input('Enter new upper bound:\n')
                     while not tflag3:
                        try: 
                           nlb,nub = float(nlb),float(nub)
                           if ((nlb < 0) | (nub > np.min(np.shape(data))) | (nub <= nlb)): raise ValueError
                           bounds_arr[2*ilf],bounds_arr[2*ilf+1] = nlb,nub
                           plot_multiple_peaks(cdat,tp,theight,apmin=apmin,apmax=apmax,maxpeaks=num_peaks)
                           for ilt in range(0,2*num_peaks): plt.axvline(bounds_arr[ilt],color='k')
                           tflag3 = True
                           inp_aps2 = raw_input("New bounds for peak %i are: (%.1f,%.1f). Are these okay? (y/n)\n"%(ilf+1,bounds_arr[2*ilf],bounds_arr[2*ilf+1]))
                        except ValueError:
                           print 'Invalid input. Bounds must be floats between 0 and %.1f.\n'%(np.min(np.shape(data)))
                           nlb = raw_input("Current bounds for peak %i are (%.1f,%.1f). Enter new lower bound:\n"%(ilf+1,bounds_arr[2*ilf],bounds_arr[2*ilf+1]))
                           nub = raw_input('Enter new upper bound:\n')
                  else:
                     print 'Invalid input\n'
                     inp_aps2 = raw_input("Are the bounds for peak %i okay? (y/n)\n"%(ilf+1))
            tflag = True
         else:
            print 'Invalid input\n'
            inp_aps = raw_input("Are these bounds okay? (y/n)\n")
   try:
      bnds_bool = (bounds_arr == np.array([0,np.min(np.shape(data))])).all()
   except AttributeError:
      bnds_bool = (bounds_arr == np.array([0,np.min(np.shape(data))]))
   if ((fitpeaks[0]) & (len(fitpeaks[fitpeaks]) == 1) & bnds_bool): 
      fitmp = False
      print 'No secondary peaks selected. Reverting to normal analysis.'
   num_peaks = len(fitpeaks[fitpeaks])
   mp_out = np.zeros((4,num_peaks))
   for impo in range(0,maxpeaks): 
      if fitpeaks[impo]: 
         inow = len(fitpeaks[0:impo+1][fitpeaks[0:impo+1]])
         mp_out[:,inow-1] = tp[inow-1]
   if output_plot != None:
      outplotname = 'bounds.%s'%output_plot
      if output_plot_dir != None: outplotname = '%s/%s'%(output_plot_dir,outplotname)
      plot_multiple_peaks(cdat,np.transpose(mp_out),theight,apmin=apmin,apmax=apmax,maxpeaks=num_peaks,plot_fits=False)
      for il in range(0,2*num_peaks): plt.axvline(bounds_arr[il],color='k')
      plt.title('Compressed Spatial Plot with Extraction Regions')
      plt.savefig(outplotname)
   if num_peaks == 1:
      return False,fixmu,tp[0],bounds_arr
   else:
      return fitmp,fixmu,mp_out,bounds_arr

#-----------------------------------------------------------------------

def find_trace(data,dispaxis="x",apmin=-4.,apmax=4.,do_plot=True,
               do_subplot=False,findmultiplepeaks=False,get_bkgd=False,get_amp=False,nofit=False):
   """
   First step in the reduction process.  
   Runs find_peak on the full 2d spectrum in order to set the initial 
    guess for trace_spectrum.
   """
   fitmp,fixmu = False,False
   if nofit: findmultiplepeaks,get_bkgd,get_amp=False,False,False
   if findmultiplepeaks:
      fitmp,fixmu,mp_vars,bounds_arr = find_multiple_peaks(data)
   if fitmp:
      p = find_peak(data,dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=do_plot,do_subplot=do_subplot,mu0=mp_vars[1],sig0=mp_vars[2])
   elif fixmu:
      p = find_peak(data,dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=do_plot,do_subplot=do_subplot,mu0=mp_vars[1],sig0=mp_vars[2],fixmu=True)
   elif nofit:
      find_peak(data,dispaxis=dispaxis,showplot=do_plot,do_subplot=do_subplot,nofit=True)
   else:
      p = find_peak(data,dispaxis=dispaxis,apmin=apmin,apmax=apmax,showplot=do_plot,do_subplot=do_subplot)
   if not nofit:
      mu0  = p[1]
      sig0 = p[2]
      bkgd = p[0]
      amp0 = p[3]
      if np.shape(bkgd) != (): bkgd = np.sum(p[0])
      if get_bkgd:
         if get_amp:
            return mu0,sig0,amp0,bkgd
         else:
            return mu0,sig0,bkgd
      elif get_amp:
         return mu0,sig0,amp0
      else:
         return mu0,sig0

#-----------------------------------------------------------------------

def fit_poly_to_trace(x, data, fitorder, data0, x_max, fitrange=None,
                      do_plot=True, markformat='bo', ylabel='Centroid of Trace',
                      title='Location of the Peak'):

   # Do a sigma clipping to reject clear outliers
   if fitrange is None:
      tmpfitdat = data
   else:
      fitmask = numpy.logical_and(x>=fitrange[0],x<fitrange[1])
      tmpfitdat = data[fitmask]
   if len(tmpfitdat[tmpfitdat != tmpfitdat[0]]) == 0:
      dmu,dsig = tmpfitdat[0],3.
      goodmask = np.arange(len(tmpfitdat))
      badmask = np.where(tmpfitdat != tmpfitdat[0])[0]
   else:
      dmu,dsig = c.sigma_clip(tmpfitdat,3.0)
      goodmask = numpy.absolute(data - dmu)<3.0*dsig
      badmask  = numpy.absolute(data - dmu)>=3.0*dsig
   dgood    = data[goodmask]
   dbad     = data[badmask]
   xgood    = x[goodmask]
   xbad     = x[badmask]

   # Fit a polynomial to the trace

   if fitrange is None:
      dpoly = numpy.polyfit(xgood,dgood,fitorder)
   else:
      fitmask = numpy.logical_and(xgood>=fitrange[0],xgood<fitrange[1])
      print fitmask
      tmpx  = xgood[fitmask]
      tmpd  = dgood[fitmask]
      dpoly = numpy.polyfit(tmpx,tmpd,fitorder)

   # Plot the results

   ymin = dmu - 4.5*dsig
   ymax = dmu + 4.5*dsig
   if do_plot:
      plt.plot(x,data,markformat)
      #plt.plot(xstep,mu,marker='o',mec='b',mfc='w',markersize=8,linestyle='')
      plt.xlabel("Pixel number in the dispersion direction")
      plt.ylabel(ylabel)
      plt.title(title)

      # Show the value from the compressed spatial profile
      plt.axhline(data0,color='k',linestyle='--')

      # Mark the bad points that were not included in the fit
      plt.plot(xbad,dbad,"rx",markersize=10,markeredgewidth=2)

      # Show the fitted function
      fitx = numpy.arange(0,x_max,0.1)
      fity = 0.0 * fitx
      for i in range(dpoly.size):
         fity += dpoly[i] * fitx**(dpoly.size - 1 - i)
      plt.plot(fitx,fity,"r")

      # Show the range of points included in the fit, if fitrange was set
      if fitrange is not None:
         plt.axvline(fitrange[0],color='k',linestyle=':')
         plt.axvline(fitrange[1],color='k',linestyle=':')
         xtmp = 0.5 * (fitrange[1] + fitrange[0])
         xerr = xtmp - fitrange[0]
         ytmp = fity.min() - 0.2 * fity.min()
         plt.errorbar(xtmp,ytmp,xerr=xerr,ecolor="g",capsize=10)
      plt.ylim(ymin, ymax)

   # Return the parameters produced by the fit
   return dpoly


#-----------------------------------------------------------------------

def trace_spectrum(data,mu0,sig0,dispaxis="x",stepsize=25,muorder=3,
   sigorder=4,fitrange=None,do_plot=True,do_subplot=False,noblankcolumns=True):
   """
   Second step in the reduction process.
   Fits a gaussian plus background to portions of the spectrum separated
   by stepsize pixels (default is 25).
   """

   # Set the dispersion axis direction
   if dispaxis == "y":
      specaxis  = 0
      spaceaxis = 1
   else:
      specaxis  = 1
      spaceaxis = 0
   if noblankcolumns:
      gfbc_all = find_blank_columns(data,output_dims=2)
      gfbc_rows = find_blank_columns(data)
   else:
      gfbc_rows = np.arange(np.shape(data)[specaxis])
      gfbc_all = np.where(data > np.min(data)-2.)
   xlength_fbc = len(gfbc_rows)
   #xlength_fbc   = data[gfbc_].shape[specaxis]
   xlength_tot   = data.shape[specaxis]

   # Define the slices through the 2D spectrum that will be used to find
   #  the centroid and width of the object spectrum as it is traced down 
   #  the chip
   xstep_all = gfbc_rows[numpy.arange(0,xlength_fbc-stepsize,stepsize)]
   xstep_fbc = numpy.arange(0,xlength_fbc-stepsize,stepsize)

   # Set up containers for mu and sigma along the trace
   if np.shape(mu0) != ():
      mu = np.zeros(((len(xstep_all)),len(mu0)))
      sigma = np.zeros(((len(xstep_all)),len(mu0)))
   else:
      mu = np.zeros(len(xstep_all))
      sigma = np.zeros(len(xstep_all))
   nsteps = numpy.arange(xstep_all.shape[0])

   # Step through the data
   print ""
   print "Running fit_trace"
   print "--------------------------------------------------------------- "
   print "Finding the location and width of the peak at %d segments of " % \
     nsteps.shape[0]
   print"   the 2D spectrum..."
   for i in nsteps:
      if(specaxis == 0):
         tmpdata = data[gfbc_rows,:][xstep_fbc[i]:xstep_fbc[i]+stepsize,:]
      else:
         tmpdata = data[:,gfbc_rows][:,xstep_fbc[i]:xstep_fbc[i]+stepsize]
      if np.shape(mu0) != ():
         ptmp = find_peak(tmpdata,dispaxis=dispaxis,showplot=False,verbose=False,mu0=mu0,sig0=sig0)
      else:
         ptmp = find_peak(tmpdata,dispaxis=dispaxis,showplot=False,verbose=False)
      mu[i]    = ptmp[1]
      sigma[i] = ptmp[2]
      if mu[i] > np.min(np.shape(data))+1: mu[i] = np.min(np.shape(data))+1
      if mu[i] < 0: mu[i] = 0
   print "   Done"

   # Fit a polynomial to the trace
   if do_plot:
      if(do_subplot):
         plt.subplot(222)
      else:
         plt.figure(2)
         plt.clf()
   print "Fitting a polynomial of order %d to the location of the trace" \
       % muorder
   if np.shape(mu0) == ():
      mupoly = fit_poly_to_trace(xstep_all,mu,muorder,mu0,xlength_tot,fitrange,do_plot=do_plot)
   else:
      plots_arr = np.array([2,5,7,9])
      mupolytmp = fit_poly_to_trace(xstep_all,mu[:,0],muorder,mu0[0],xlength_tot,fitrange,do_plot=do_plot)
      mupoly = np.zeros((len(mu0),len(mupolytmp)))
      mupoly[0] = mupolytmp
      for impt in range(1,len(mu0)):
         if do_plot:
            if do_subplot:
               plt.figure(plots_arr[impt])
               plt.clf()
               plt.subplot(222)
         mupoly[impt] = fit_poly_to_trace(xstep_all,mu[:,impt],muorder,mu0[impt],xlength_tot,fitrange,do_plot=do_plot)
      mupoly = np.transpose(mupoly)
   # Fit a polynomial to the width of the trace
   if(do_subplot):
      plt.figure(2)
      plt.subplot(223)
   else:
      plt.figure(3)
      plt.clf()
   print "Fitting a polynomial of order %d to the width of the trace" % sigorder
   if np.shape(mu0) == ():
      sigpoly = fit_poly_to_trace(xstep_all,sigma,sigorder,sig0,xlength_tot,fitrange,markformat='go',title='Width of Peak',ylabel='Width of trace (Gaussian sigma)',do_plot=do_plot)
   else:
      plots_arr = np.array([2,5,7,9])
      sigpolytmp = fit_poly_to_trace(xstep_all,sigma[:,0],sigorder,sig0[0],xlength_tot,fitrange,markformat='go',title='Width of Peak',ylabel='Width of trace (Gaussian sigma)',do_plot=do_plot)
      sigpoly = np.zeros((len(mu0),len(mupolytmp)))
      sigpoly[0] = sigpolytmp
      for ispt in range(1,len(mu0)):
         if do_plot:
            if do_subplot:
               plt.figure(plots_arr[ispt])
               plt.subplot(223)
         sigpoly[ispt] = fit_poly_to_trace(xstep_all,sigma[:,ispt],sigorder,sig0[ispt],xlength_tot,fitrange,markformat='go',title='Width of Peak',ylabel='Width of trace (Gaussian sigma)',do_plot=do_plot)
      sigpoly = np.transpose(sigpoly)

   # Return the fitted parameters
   return mupoly,sigpoly

#-----------------------------------------------------------------------

def extract_spectrum(data,mupoly,sigpoly,dispaxis="x",apmin=-4.,apmax=4.,
                     weight='gauss', sky=None, gain=1.0, rdnoise=0.0, 
                     do_plot=True, do_subplot=False, outfile=None, weighted_var=True,var_in=None,bkgd=None,amp0=None):
   """
   Third step in reduction process.
   This function extracts a 1D spectrum from the input 2D spectrum (data)
   It uses the information about the trace that has been generated by
   the trace_spectrum function.  In particular, it takes the two
   polynomials generated by trace_spectrum as the inputs mupoly and sigpoly.
   """

   # Set the dispersion axis direction
   if dispaxis == "y":
      specaxis  = 0
      spaceaxis = 1
   else:
      specaxis  = 1
      spaceaxis = 0

   # Set the wavelength axis
   pix = numpy.arange(data.shape[specaxis])

   # Set the fixed mu and sigma for the Gaussian fit at each point in the
   #  spectrum, using the input polynomials

   fitx = numpy.arange(data.shape[specaxis]).astype(numpy.float32)
   if np.shape(mupoly[0]) == ():
      mu = 0.0 * fitx
      for i in range(mupoly.size):
         mu += mupoly[i] * fitx**(mupoly.size - 1 - i)
      sig = 0.0 * fitx
      for i in range(sigpoly.size):
         sig += sigpoly[i] * fitx**(sigpoly.size - 1 - i)
   else:
      mu = np.zeros((len(mupoly[0]),len(fitx)))
      for i in range(len(mupoly[0])):
         mu += np.transpose(np.array([mupoly[i]])) * fitx**(len(mupoly[0])- 1 - i)
      sig = np.zeros((len(mupoly[0]),len(fitx)))
      for i in range(len(mupoly[0])):
         sig += np.transpose(np.array([sigpoly[i]])) * fitx**(len(mupoly[0]) - 1 - i)
      mu,sig = np.transpose(mu),np.transpose(sig)
      

   # Set up the containers for the amplitude and variance of the spectrum
   #  along the trace
   amp = 0.0 * pix
   var = 0.0 * pix
   if np.shape(mupoly[0]) != ():
      amp,var = np.zeros((len(pix),len(mupoly[0]))),np.zeros((len(pix),len(mupoly[0])))
   # Step through the data
   print ""
   print "Extracting the spectrum..."
   if not weighted_var: tmpvar = None
   for i in pix:
      #print pix[i], mu[i], sig[i]
      if specaxis == 0:
         tmpdata = data[i,:]
         if weighted_var: tmpvar = var_in[i,:]
      else:
         tmpdata = data[:,i]
         if weighted_var: tmpvar = var_in[:,i]
      if ((sky == None) | weighted_var):
         skyval = None
      else:
         skyval = sky[i]
      #print tmpdata,mu[i],apmin,apmax,sig[i],gain,rdnoise,skyval,weight,tmpvar,bkgd,amp0
      amp[i],var[i] = extract_wtsum_col(tmpdata,mu[i],apmin,apmax,sig=sig[i],gain=gain,rdnoise=rdnoise,sky=skyval,weight=weight,weighted_var=True,var_in=tmpvar,bkgd=bkgd,amp0=amp0)
   print "   Done"

   # Plot the extracted spectrum
   if do_plot:
      print ""
      print "Plotting the spectrum"
      if np.shape(mupoly[0]) == ():
         if(do_subplot):
            plt.subplot(224)
         else:
            plt.figure(4)
            plt.clf()
         plot_spectrum_array(pix,amp,title='Extracted spectrum (pixels)',xlabel='Pixel number in the dispersion direction')
      else:
         plots_arr = np.array([2,5,7,9])
         for ipex in range(0,len(mupoly[0])):
            if(do_subplot):
               plt.figure(plots_arr[ipex])
               plt.subplot(224)
            else:
               plt.figure(4)
               plt.clf()
            plot_spectrum_array(pix,amp[:,ipex],title='Extracted spectrum (pixels)',xlabel='Pixel number in the dispersion direction')

   # Save the extracted spectrum
   if outfile is not None:
      save_spectrum(outfile,pix,amp,var)

   # Return the extracted spectrum
   return amp,var

#-----------------------------------------------------------------------

def resample_spec(w, spec, owave=None):
   """
   Given an input spectrum, represented by wavelength values (w) and fluxes
    (spec), resample onto a linearized wavelength grid.  The grid can either
    be defined by the input wavelength range itself (the default) or by
    a wavelength vector that is passed to the function.
   """

   if owave is None:
      w0 = w[0]
      w1 = w[-1]
      owave = numpy.linspace(w0,w1,w.size)
   
   if np.shape(spec[0]) == ():
      specmod = interpolate.splrep(w,spec)
      outspec = interpolate.splev(owave,specmod)
   else:
      outspec = np.zeros(np.shape(spec))
      for ios in range(0,len(spec[0])):
         specmod = interpolate.splrep(w,spec[:,ios])
         outspec[:,ios] = interpolate.splev(owave,specmod)

   return owave,outspec

#-----------------------------------------------------------------------

def combine_spectra(txt_files,outfile):
   """
   Given the input spectra, stored as text files (e.g., "vega*txt"), reads in 
   the files and does an inverse-variance weighted combination of the counts 
   columns.
   """

   file_list = glob.glob(txt_files)

   """ Setup """
   tmpdat = numpy.loadtxt(file_list[0])
   wavelength = tmpdat[:,0].copy()
   wtflux = wavelength * 0.0
   wtsum  = wavelength * 0.0

   """ Create the weighted sum """
   print ""
   for f in file_list:
      print "Reading data from file %s" % f 
      tmpdat = numpy.loadtxt(f)
      wt = 1.0 / (tmpdat[:,2])
      wtflux += wt * tmpdat[:,1]
      wtsum += wt

   """ 
   Normalize the flux, and calculate the variance of the coadded spectrum.
   Note that the equation below for the variance only works for the case
    of inverse variance weighting.
   """
   outspec = wtflux / wtsum
   outvar  = 1.0 / wtsum

   """ Plot the combined spectrum """
   plot_spectrum_array(wavelength,outspec,outvar,xlabel="Pixels",
                       title="Combined spectrum")

   print "Saving the combined spectrum"
   save_spectrum(outfile,wavelength,outspec,outvar)

#-----------------------------------------------------------------------

def plot_sky(infile):
   """
   Given an input 2-dimensional fits file for which the sky has NOT been 
   subtracted, this function will take the median along the spatial axis
   to produce a sky spectrum. 

   *** NB: Right now this ASSUMES that the dispersion is in the x direction ***
   """

   data = pyfits.getdata(infile)
   sky = numpy.median(data,axis=0)
   pix = numpy.arange(sky.size)
   plot_spectrum_array(pix,sky,xlabel='Pixels',title='Sky Spectrum')

#-----------------------------------------------------------------------

def make_sky_model(wavelength, smoothKernel=25., verbose=True):
   """
   Given an input wavelength vector, creates a smooth model of the
   night sky emission that matches the wavelength range and stepsize
   of the input vector.
   """

   # Get info from input wavelength vector
   wstart = wavelength.min()
   wend = wavelength.max()
   disp = wavelength[1] - wavelength[0]
   if verbose:
      print "Making model sky"
      print "--------------------------------------"
      print "Model starting wavelength: %f" % wstart
      print "Model ending wavelength:   %f" % wend
      print "Model dispersion:          %f" % disp

   # Use NIR sky model if starting wavelength is redder than 9000 Angstrom
   if wstart >= 9000.:
      # Read in skymodel, which is in a B-spline format
      skymodel_file = '/Users/cdf/Code/python/nirspec/nirspec_skymodel.dat'
      skymodel = numpy.load(skymodel_file)

      # Resample and smooth the model spectrum
      wave = numpy.arange(wstart,wend,0.2)
      tmpskymod = interpolate.splev(wave,skymodel)
      tmpskymod = ndimage.gaussian_filter(tmpskymod,smoothKernel)

      # Create a B-spline representation of the smoothed curve for use in
      #  the wavecal optimization
      model = interpolate.splrep(wave,tmpskymod)

      # Finally use the initial guess for the dispersion and evaluate the
      #  model sky at those points, using the B-spline model
      skymod = interpolate.splev(wavelength,model)

      # Clean up and return
      del skymodel,tmpskymod,wave
      return skymod

   else:
      print ""
      print "Optical sky model is not yet implemented."
      print ""
      return None


#-----------------------------------------------------------------------

def apply_wavecal(infile,outfile,lambda0,dlambda):
   """
   Given an input file containing 2 columns (x and flux), and the y-intercept
   and slope of the x-lambda transformation, convert x to wavelength units
   and write the output as outfile.
   """

   """ Read the input file """
   xspec = numpy.loadtxt(infile)

   """ Convert x from pixels to wavelength units """
   wavelength = lambda0 + dlambda * xspec[:,0]

   """ Plot the results """
   plot_spectrum_array(wavelength,xspec[:,1])
   plt.title("Wavelength-calibrated spectrum")

   """ Save the results """
   save_spectrum(outfile,wavelength,xspec[:,1])

#-----------------------------------------------------------------------

def check_wavecal(infile, informat='text', modsmoothkernel=25.):
   """
   Plots the wavelength-calibrated sky information from the input file
   on top of a smoothed model of the night sky emission so that
   the quality of the wavelength calibration can be evaluated.

   The input file can either be a fits file (fileformat='fits') that has
   been produced by the niredux function or a text spectrum containing
   three columns (wavelength, flux, variance).  The variance spectrum in
   this case should contain clear sky line features if the exposure length
   was long enough.
   """

   """ Read in the observed sky spectrum """
   if informat=='fits':
      hdulist = pyfits.open(infile)
      varspec = hdulist[1].data.copy()
      skyobs  = n.sqrt(n.median(varspec,axis=0))
      skylab  = "RMS Spectrum"
      hdr = hdulist[0].header
      crval1  = hdr['crval1']
      crpix1  = hdr['crpix1']
      cd11    = hdr['cd1_1']
      waveobs = 1.0* n.arange(varspec.shape[1])
      waveobs *= cd11
      waveobs += crval1
   elif informat=='fitsold':
      hdulist = pyfits.open(infile)
      waveobs = hdulist[1].data.copy()
      skyobs  = hdulist[2].data.copy()
      skylab  = "Observed Sky"
   else:
      try:
         waveobs,varspec = numpy.loadtxt(infile,unpack=True,usecols=(0,2))
         skyobs = n.sqrt(varspec)
      except:
         print ""
         print "Cannot get variance spectrum from input text file %s" % infile
         print ""
         return
      skylab = "RMS Spectrum"

   """ Create the sky spectrum, with the appropriate smoothing """
   print ""
   skymod = make_sky_model(waveobs,modsmoothkernel)

   """ 
   Scale the sky spectrum to roughly be 75% of the amplitude of the observed
   spectrum
   """

   deltamod = skymod.max() - skymod.min()
   deltaobs = skyobs.max() - skyobs.min()
   skymod *= 0.75*deltaobs/deltamod
   skymod += skyobs.mean() - skymod.mean()

   """ Make the plot """
   wrange = waveobs.max() - waveobs.min()
   xmin = waveobs.min() - 0.05*wrange
   xmax = waveobs.max() + 0.05*wrange
   plt.plot(waveobs,skyobs,'k',ls='steps', label=skylab)
   plt.plot(waveobs,skymod,'r',ls='steps', label='Model sky')
   plt.legend()
   plt.xlim(xmin,xmax)

   del waveobs
   del skyobs
   del skymod

#-----------------------------------------------------------------------

def planck_spec(wavelength, T=1.0e4, waveunit='Angstrom'):
   """
   Given a wavelength vector and an input temperture, generates a thermal
    spectrum over the input wavelength range.
   The input spectrum is B_lambda(T) and NOT B_nu(T)
   """

   # Define the constants in the Planck function in SI units
   c = 3.0e8
   h = 6.626e-34
   k = 1.38e-23

   # Convert the wavelength into meters (default assumption is that the
   #  input wavelength is in Angstroms
   wtmp = wavelength.copy()
   if waveunit[0:6].lower()=='micron':
      print "Converting wavelength from microns to meters"
      wtmp *= 1.0e-6
   elif waveunit[0:5].lower()=='meter':
      wtmp *= 1.0
   else:
      print "Converting wavelength from Angstroms to meters"
      wtmp *= 1.0e-10

   # Generate the Planck function, and then scale it so that its mean matches
   #  the mean of the observed spectrum, just for ease in plotting.
   from math import e
   denom = e**(h * c /(wtmp * k * T)) - 1.0
   B_lam = 2.0 * h * c**2 / (wtmp**5 * denom)

   return B_lam

#-----------------------------------------------------------------------

def response_ir(infile,outfile,order=6,fitrange=None,filtwidth=9):
   """
   Calculates the response function for an IR spectral setup using observations
   of a standard star.  The assumption is that the standard is a hot
   star (A or B), and therefore its spectrum is just a power law in the NIR.
   The steps are:
     (1) Divide the model spectrum by the observed spectrum
     (2) Fit a polynomial to the result
     (3) Write out the result to the output file
   The response function in the output file can then be multiplied by other
   spectra to do a response correction (an approximation of flux calibration).

   Inputs:
      infile:    Input file containing observed spectrum of the hot star
                  This file should have 3 columns (wavelength, flux, variance)
      outfile:   Output file that will contain the response function
      order:     Order of polynomial fit (default = 6)
      fitrange:  A list of 2-element lists, where each of the smaller lists
                  contains the starting and ending values for a range of
                  good data to include in the fit.
                  E.g., fitrange=[[20150.,21500.],[22000.,25000.]]
                  The default (fitrange=None) uses the full wavelength range
                  in the fit.
      filtwidth: Width of box used in the maximum filtering step, which is
                  used to minimize the number of absorption lines in the
                  spectrum before fitting a low-order polynomial to the result
                  (default = 9)
   """

   # Read the input spectrum
   wave,fluxobs,var = read_spectrum(infile)
   rms = numpy.sqrt(var)

   # Generate the thermal spectrum and normalize it
   B_lam = planck_spec(wave)
   B_lam *= fluxobs.mean() / B_lam.mean()

   # The features in the observed spectrum that deviate from a thermal
   #  spectrum should only be absorption lines.  Therefore, run a maximum
   #  filter

   flux = ndimage.filters.maximum_filter(fluxobs,filtwidth)

   # Calculate the observed response function
   respobs = B_lam / flux

   # Show some plots
   plt.figure(1)
   plt.clf()
   plt.plot(wave,fluxobs)
   plt.plot(wave,B_lam)
   plt.plot(wave,flux)
   plt.plot(wave,rms)
   plt.figure(2)
   plt.clf()
   plt.plot(wave,respobs)

   # Define the spectral range to be included in the fit
   if fitrange is not None:
      mask = numpy.zeros(respobs.size,dtype=numpy.bool)
      fitr = numpy.atleast_2d(numpy.asarray(fitrange))
      for i in range(fitr.shape[0]):
         wmask = numpy.logical_and(wave>fitr[i,0],wave<fitr[i,1])
         mask[wmask] = True
      wavegood = wave[mask]
      respgood = respobs[mask]
   else:
      wavegood = wave
      respgood = respobs

   # Fit a polynomial to the observed response function
   fpoly = numpy.polyfit(wavegood,respgood,order)
   print ""
   print "Fit a polynomial of order %d to curve in Figure 2." % order
   print "Resulting coefficients:"
   print "-----------------------"
   print fpoly

   # Convert polynomial into a smooth response function
   p = numpy.poly1d(fpoly)
   resp = p(wave)

   # Add the smooth response to the plot and show corrected curve
   plt.plot(wave,resp,'r')
   fc = fluxobs * resp
   plt.figure(3)
   plt.clf()
   plt.plot(wave,fc)
   plt.plot(wave,B_lam)

   # Write smooth response to output file
   out = numpy.zeros((wave.size,2))
   out[:,0] = wave
   out[:,1] = resp
   numpy.savetxt(outfile,out,'%8.3f  %.18e')

#-----------------------------------------------------------------------

def response_correct(infile, respfile, outfile):
   """
   Applies a response correction, calculated previously by response_ir
   or another function, to the input file.  The output is placed in
   outfile.

   Inputs:
      infile:   Input spectrum
      respfile: Response correction spectrum
      outfile:  Output spectrum
   """

   # Read input files
   try:
      w,f,v = numpy.loadtxt(infile,unpack=True)
   except:
      print ""
      print "ERROR: response_correct.  Unable to read input spectrum %s" \
          % infile
      return
   try:
      wr,resp = numpy.loadtxt(respfile,unpack=True)
   except:
      print ""
      print "ERROR: response_correct.  Unable to read response spectrum %s" \
          % respfile
      return

   # Apply the response correction and save the spectrum
   f *= resp
   v *= resp**2
   save_spectrum(outfile,w,f,v)

#-----------------------------------------------------------------------

def normalize(infile,outfile,order=6,fitrange=None,filtwidth=11):
   """
   Normalizes a spectrum by fitting to the continuum and then dividing the
    input spectrum by the fit.

   Inputs:
      infile:    File containing the input spectrum
                  This file should have 3 columns (wavelength, flux, variance)
      outfile:   Output file that will contain the normalized spectrum
      order:     Order of polynomial fit (default = 6)
      fitrange:  A list of 2-element lists, where each of the smaller lists
                  contains the starting and ending values for a range of
                  good data to include in the fit.
                  E.g., fitrange=[[20150.,21500.],[22000.,25000.]]
                  The default (fitrange=None) uses the full wavelength range
                  in the fit.
      filtwidth: Width of box used in the boxcar smoothing step, which is
                  used to minimize the number of outlier points in the input
                  spectrum before fitting the polynomial to the spectrum
                  (default = 9)
   """

   # Read the input spectrum
   wave,fluxobs,var = read_spectrum(infile)
   rms = numpy.sqrt(var)

   # Try to minimize outliers due to both emission and absorption
   #  lines and to cosmetic features (cosmic rays, bad sky-line subtraction).
   #  Do this by doing a inverse-variance weighted boxcar smoothing.

   wt = 1.0 / var
   yin = wt * fluxobs
   flux = ndimage.filters.uniform_filter(yin,filtwidth)
   flux /= ndimage.filters.uniform_filter(wt,filtwidth)

   # Show some plots
   plt.figure(1)
   plt.clf()
   plt.plot(wave,fluxobs)
   plt.plot(wave,flux,'r')

   # Define the spectral range to be included in the fit
   if fitrange is not None:
      mask = numpy.zeros(flux.size,dtype=numpy.bool)
      fitr = numpy.atleast_2d(numpy.asarray(fitrange))
      for i in range(fitr.shape[0]):
         wmask = numpy.logical_and(wave>fitr[i,0],wave<fitr[i,1])
         mask[wmask] = True
      wavegood = wave[mask]
      fluxgood = flux[mask]
   else:
      wavegood = wave
      fluxgood = flux

   # Fit a polynomial to the observed response function
   fpoly = numpy.polyfit(wavegood,fluxgood,order)
   print ""
   print "Fit a polynomial of order %d to the red curve in Figure 1." % order
   print "Resulting coefficients:"
   print "-----------------------"
   print fpoly

   # Convert polynomial into a smooth response function
   p = numpy.poly1d(fpoly)
   cfit = p(wave)

   # Add the smooth response to the plot and show corrected curve
   plt.plot(wave,cfit,'k')
   fc = fluxobs / cfit
   vc = var / cfit**2
   plt.figure(2)
   plt.clf()
   plt.plot(wave,fc)

   # Write normalized spectrum to output file
   save_spectrum(outfile,wave,fc,vc)

#-----------------------------------------------------------------------

def mark_spec_emission(z, w=None, f=None, labww=20., labfs=12, ticklen=0.,showz=True):
   """
   Marks the location of expected emission lines in a spectrum, given
    a redshift (z).  The default behavior is just to mark the lines with
    a vertical dashed line from the top to the bottom of the plot.  
    However, if the wavelength (w) and flux (f) vectors are passed to
    the function, then the lines are marked with short vertical tick marks.

   Inputs:
      z       - redshift to be marked
      w       - [OPTIONAL] wavelength array. Only used if short-tick line 
                marker style is desired.
      f       - [OPTIONAL] flux array. Only used if short-tick line marker
                style is desired.
      labww   - width in pixels of the window used to set the vertical
                location of the tickmark (location is set from the maximum
                value within the window).
      labfs   - font size for labels, in points
      ticklen - override of auto-determination of tick length if > 0
   """

   linelist = numpy.array([
         1216.,1549.,1909.,2800.,
         3726.03,3728.82,4861.33,4962.,5007.,#5199.,6300.,
         6548.,6562.8,
         6583.5,6716.4,6730.8,10900.,12800.,18700.])
   linename = [
      "Ly-alpha","CIV","CIII]","MgII",
      "[OII]","",'H-beta','[OIII]','[OIII]',#'[NI]','[OI]',
      '[NII]','H-alpha',
      '[NII]','[SII]','','Pa-gamma','Pa-beta','Pa-alpha']

   lineinfo = n.array([\
      ("Ly-alpha",  1216.,   r"Ly $\alpha$",1,True),\
      ("C IV",      1549.,   "C IV",        1,True),\
      ("C III]",    1909.,   "C III]",      1,True),\
      ("Mg II",     2800.,   "Mg II",       1,True),\
      ("[O II]",    3726.03, "[O II]",      1,True),\
      ("[O II]",    3728.82, "[O II]",      1,False),\
      ("H-beta",    4861.33, r"H$\beta$",    1,True),\
      ("[O III]",   4962.,   "[O III]",     1,False),\
      ("[O III]",   5007.,   "[O III]",     1,True),\
      #("[N I]",     5199.,   "[N I]",       1,False),\
      ("[O I]",     6300.,   "[O I]",       1,True),\
      ("[N II]",    6548.,   "[N II]",      1,True),\
      ("H-alpha",   6562.8,  r"H$\alpha$",  1,True),\
      ("[N II]",    6583.5,  "[N II]",      1,True),\
      ("[S II]",    6716.4,  "[S II]",      1,False),\
      ("[S II]",    6730.8,  "[S II]",      1,True),\
      ("Pa-gamma", 10900.,   r"Pa $\gamma$",1,True),\
      ("Pa-beta",  12800.,   r"Pa $\beta$", 1,True),\
      ("Pa-alpha", 18700.,   r"Pa $\alpha$",1,True)\
      ],\
      dtype=[('name','S10'),('wavelength',float),('label','S10'),\
             ('dir',int),('plot',bool)]\
      )
   #print lineinfo


   """ Get current display limits """
   if w is None:
      lammin, lammax = plt.xlim()
   else:
      if f is None:
         print ""
         print "ERROR: mark_spec_emission. w is set but f isn't."
         print "Changing to default marking style"
         print ""
         w = None
      else:
         lammin,lammax = w.min(),w.max()
         if plt.xlim()[0] > lammin: lammin = plt.xlim()[0]
         if plt.xlim()[1] < lammax: lammax = plt.xlim()[1]
         dlam = w[1] - w[0]
         ff = f[(w>=plt.xlim()[0]) & (w<=plt.xlim()[1])]
         fluxdiff = ff.max() - ff.min()
         dlocwin = labww / 2.
         #if ticklen == 0.:
            #ticklen = 0.1 * fluxdiff


   """ Only mark lines within current display range """
   zlines = (z+1.0) * lineinfo['wavelength']
   print ""
   print "Line      lambda_obs"
   print "--------  -----------"
   for i in range(len(lineinfo)):
      print "%-9s %8.2f" % (lineinfo['name'][i],zlines[i])
   mask = numpy.logical_and(zlines>lammin,zlines<lammax)
   tmplines = lineinfo[mask]
   if (len(tmplines) > 0):
      tmpfmax,xarr = np.zeros(0),np.zeros(0)
      for i in tmplines:
         x = i['wavelength']*(z+1.0)
         xarr = np.append(xarr,x)
         if w is not None and f is not None:
            tmpmask = numpy.where((w>=x-dlocwin*dlam) &(w<=x+dlocwin*dlam))
            tmpfmax = np.append(tmpfmax,f[tmpmask].max())
      #tickstarts = tmpfmax - 0.25*(tmpfmax-plt.ylim()[1]) 
      #labstarts  = tmpfmax - 0.4*(tmpfmax-plt.ylim()[1])
      tmpticklens = -0.25*(tmpfmax-plt.ylim()[1])
      if len(tmpticklens) > 0:
         if ticklen == 0.:
            tmpticklen = np.max([np.min([(plt.ylim()[1]-plt.ylim()[0])/30.,np.min([tmpticklens[tmpticklens > 0]])]),(plt.ylim()[1]-plt.ylim()[0])/40.])
         else:
            tmpticklen = ticklen
      for i in range(0,len(tmplines)):
         if w is not None and f is not None:
            #tickstart = tmpfmax + 0.05 * fluxdiff
            #labstart  = tickstart + 1.3 * ticklen
            #if tmpfmax[i] < plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])/15.:
            tickstart = tmpfmax[i]+0.5*tmpticklen
            labstart = tmpfmax[i]+2*tmpticklen
            if plt.ylim()[1]-tmpfmax[i] > 3*tmpticklen:
               plt.plot([xarr[i],xarr[i]],[tickstart,tickstart+tmpticklen],'k')
               if tmplines[i]['plot']:
                  plt.text(xarr[i],labstart,tmplines[i]['label'],color='k',rotation='vertical',ha='center',va='bottom',fontsize=labfs)
            elif tmpfmax[i] < plt.ylim()[1]:
               plt.axvline(xarr[i],linestyle='--',color='k',lw=1)
               #print i['label'],tickstart,labstart,tmpticklen,tmpfmax[i]
         else:
            plt.axvline(xarr[i],linestyle='--',color='k',lw=1)
   if showz: 
      plt.text(plt.xlim()[0]+0.05*(plt.xlim()[1]-plt.xlim()[0]),plt.ylim()[1]-0.05*(plt.ylim()[1]-plt.ylim()[0]),'z = %5.3f'%z,color='k',rotation='horizontal',ha='left',va='center',fontsize=labfs+4)



#-----------------------------------------------------------------------

def mark_spec_absorption(z, w=None, f=None, labww=20., labfs=12, ticklen=0.,showz=True):
   """
   Marks the location of expected absorption lines in a spectrum, given
    a redshift (z).  The default behavior is just to mark the lines with
    a vertical dashed line from the top to the bottom of the plot.  
    However, if the wavelength (w) and flux (f) vectors are passed to
    the function, then the lines are marked with short vertical tick marks.

   Inputs:
      z       - redshift to be marked
      w       - [OPTIONAL] wavelength array. Only used if short-tick line 
                marker style is desired.
      f       - [OPTIONAL] flux array. Only used if short-tick line marker
                style is desired.
      labww   - width in pixels of the window used to set the vertical
                location of the tickmark (location is set from the minimum
                value within the window).
      labfs   - font size for labels, in points
      ticklen - override of auto-determination of tick length if > 0
   """

   linelist = numpy.array([
         3883,3933.667,3968.472,4101,4305,4340,4383,4455,4531,4861,5176,5893])
   linename = [
      "CN bandhead","CaII K","CaII H","H-delta","G-band","H-gamma","Fe4383","Ca4455","Fe4531","H-beta","Mg I (b)","Na I (D)"]

   lineinfo = n.array([\
       ("CN bandhead",   3883,       "CN red",1,True),\
       ("CaII K",        3933.667,   "CaII K",1,True),\
       ("CaII H",        3968.472,   "CaII H",1,True),\
       ("H-delta",       4101,       r"H$\delta$",1,True),\
       ("G-band",        4305,       "G-band",1,True),\
       ("H-gamma",       4340,       r"H$\gamma$",1,True),\
       ("Fe4383",        4383,       "Fe4383",1,True),\
       ("Ca4455",        4455,       "Ca4455",1,True),\
       ("Fe4531",        4531,       "Fe4531",1,True),\
       ("H-beta",        4861,       r"H$\beta$",1,True),\
       ("Mg I (b)",      5176,       "Mg b",1,True),\
       ("Na I (D)",      5893,       "Na D",1,True)\
      ],\
      dtype=[('name','S10'),('wavelength',float),('label','S10'),\
             ('dir',int),('plot',bool)]\
      )
   #print lineinfo

   """ Get current display limits """
   if w is None:
      lammin, lammax = plt.xlim()
   else:
      if f is None:
         print ""
         print "ERROR: mark_spec_emission. w is set but f isn't."
         print "Changing to default marking style"
         print ""
         w = None
      else:
         lammin,lammax = w.min(),w.max()
         if plt.xlim()[0] > lammin: lammin = plt.xlim()[0]
         if plt.xlim()[1] < lammax: lammax = plt.xlim()[1]
         dlam = w[1] - w[0]
         ff = f[(w>=plt.xlim()[0]) & (w<=plt.xlim()[1])]
         fluxdiff = ff.max() - ff.min()
         dlocwin = labww / 2.
         #if ticklen == 0.:
            #ticklen = 0.1 * fluxdiff


   """ Only mark lines within current display range """
   zlines = (z+1.0) * lineinfo['wavelength']
   print ""
   print "Line      lambda_obs"
   print "--------  -----------"
   for i in range(len(lineinfo)):
      print "%-9s %8.2f" % (lineinfo['name'][i],zlines[i])
   mask = numpy.logical_and(zlines>lammin,zlines<lammax)
   tmplines = lineinfo[mask]
   if (len(tmplines) > 0):
      tmpfmin,xarr = np.zeros(0),np.zeros(0)
      for i in tmplines:
         x = i['wavelength']*(z+1.0)
         xarr = np.append(xarr,x)
         if w is not None and f is not None:
            tmpmask = numpy.where((w>=x-dlocwin*dlam) &(w<=x+dlocwin*dlam))
            tmpfmin = np.append(tmpfmin,f[tmpmask].min())
      #tickstart = tmpfmin - 0.2*(tmpfmin-plt.ylim()[0]) 
      #labstart  = tmpfmin - 0.4*(tmpfmin-plt.ylim()[0])
      tmpticklens = 0.25*(tmpfmin-plt.ylim()[0])
      if len(tmpticklens) > 0:
         if ticklen == 0.: 
            tmpticklen = np.max([np.min([(plt.ylim()[1]-plt.ylim()[0])/30.,np.min([tmpticklens[tmpticklens > 0]])]),(plt.ylim()[1]-plt.ylim()[0])/40.])
         else:
            tmpticklen = ticklen
      for i in range(0,len(tmplines)):
         if w is not None and f is not None:
            tickstart = tmpfmin[i]-0.5*tmpticklen
            labstart = tmpfmin[i]-2*tmpticklen
            if tmpfmin[i]-plt.ylim()[0] > 3*tmpticklen:
               plt.plot([xarr[i],xarr[i]],[tickstart-tmpticklen,tickstart],'k')
               #print i['label'],tickstart,labstart,tmpticklen,tmpfmin
               if tmplines[i]['plot']:
                  plt.text(xarr[i],labstart,tmplines[i]['label'],color='k',rotation='vertical',ha='center',va='top',fontsize=labfs)
            elif tmpfmin[i] > plt.ylim()[0]:
               plt.axvline(xarr[i],linestyle='--',color='k',lw=1)
               #print i['label'],tickstart,labstart,tmpticklen,tmpfmin
         else:
            plt.axvline(xarr[i],linestyle='--',color='k',lw=1)
   if showz: 
      plt.text(plt.xlim()[0]+0.05*(plt.xlim()[1]-plt.xlim()[0]),plt.ylim()[1]-0.05*(plt.ylim()[1]-plt.ylim()[0]),'z = %5.3f'%z,color='k',rotation='horizontal',ha='left',va='center',fontsize=labfs+4)


#-----------------------------------------------------------------------


def plot_atm_trans(w, fwhm=15., flux=None, scale=1.05, offset=0.0,
                   color='g', linestyle='-', return_atm=False):
   """
   Given an input spectrum, represented by the wavelength (w) and flux (spec)
   vectors, and a rough fwhm (in Angstrom), smooths and resamples the 
   atmospheric transmission spectrum for the NIR and plots it.
   """

   """ Read in the atmospheric transmission data"""
   print ""
   atm_filename = \
       '/Users/cdf/Projects/Active/nirspec_redux/mk_atm_trans_zm_10_10.dat'
   print "Loading atmospheric data from %s" % atm_filename
   atmwave,atmtrans = numpy.loadtxt(atm_filename,unpack=True)
   atmwave *= 1.0e4

   """ Only use the relevant part of the atmospheric transmission spectrum"""
   mask = numpy.where((atmwave>=w.min())&(atmwave<=w.max()))
   watm = atmwave[mask]
   atm  = atmtrans[mask]
   del atmwave
   del atmtrans

   """ Smooth the spectrum """
   atm = ndimage.gaussian_filter(atm,fwhm)

   """ Resample the smoothed spectrum """
   watm,atm = resample_spec(watm,atm,w)

   """ If an input spectrum has been given, then rescale the atm spectrum """
   if flux is not None:
      atm *= scale * flux.max()
   else:
      atm *= scale

   """ Add any requested vertical offset """
   atm += offset

   """ Plot the results """
   ls = "steps%s" % linestyle
   plt.plot(watm,atm,color,ls=ls)

   del watm
   if return_atm:
      return atm
   else:
      del atm

#-----------------------------------------------------------------------

def smooth_boxcar(infile, filtwidth, outfile=None, varwt=True, title='Smoothed Spectrum',line=1,hasvar=True,output=False,clear=False):
   """
   Does a boxcar smooth of an input spectrum.  The default is to do
   inverse variance weighting, using the variance encoded in the third column
   of the input spectrum file.
   The other default is not to write out an output file.  This can be
   changed by setting the outfile parameter.
   """

   """ Read the input spectrum """
   if clear: plt.clf()
   inspec = numpy.loadtxt(infile)
   wavelength = inspec[:,0]
   if line == 1:
      influx = inspec[:,1]
      if(varwt):
         if(inspec.shape[1] < 3):
            print ""
            print "ERROR: Inverse variance weighting requested, but input file"
            print "       has fewer than 3 columns (ncol = %d)" % inspec.shape[1]
            return
         else:
            wt = np.zeros(len(inspec[:,2]))
            wt[inspec[:,2]!=0] = 1.0 / inspec[:,2][inspec[:,2]!=0]
      else:
         wt = 0.0 * influx + 1.0
   else:
      if hasvar:
         influx = inspec[:,2*line-1]
         if varwt:
            wt = np.zeros(len(inspec[:,2*line]))
            wt[inspec[:,2*line]!=0] = 1.0 / inspec[:,2*line][inspec[:,2*line]!=0]
      else:
         influx = inspec[:,line]

   """ Smooth spectrum """
   yin = wt * influx
   outflux = ndimage.filters.uniform_filter(yin,filtwidth)
   wt_tmp = wt.copy()
   wt_tmp[wt == 0] = 0.00001*np.min(wt[wt!=0])
   outflux /= ndimage.filters.uniform_filter(wt_tmp,filtwidth)
   if varwt:
      outvar = (filtwidth * ndimage.filters.uniform_filter(wt,filtwidth))
      outvar[(influx == 0) & (outflux < np.min(np.fabs(influx[influx!=0]))/filtwidth)] *= 0.
      outvar[outvar!=0] = 1.0 / outvar[outvar!=0]

   """ Plot the smoothed spectrum """
   if varwt:
      plot_spectrum_array(wavelength,outflux,outvar,title=title)
   else:
      plot_spectrum_array(wavelength,outflux,title=title)

   """ Save the output file if desired """
   if(outfile):
      print "Saving smoothed spectrum to %s" % outfile
      if varwt:
         save_spectrum(outfile,wavelength,outflux,outvar)
      else:
         save_spectrum(outfile,wavelength,outflux)
      print ""
   if output: 
      if varwt:
         return wavelength,outflux,outvar
      else:
         return wavelength,outflux

#-----------------------------------------------------------------------

def calc_lineflux(wavelength,flux,bluemin,bluemax,redmin,redmax,var=None,
                  showsub=False):
   """
   Given vectors of flux and wavelength, interactively calculates the integrated
   flux in an emission line.  The user enters the wavelength ranges to use
   for the continuum on both the blue (bluemin and bluemax) and red (redmin
   and redmax) sides of the line.  The function will do a first order fit 
   (i.e., a line) to the continuum using these ranges, subtract the continuum
   from the data, and then numerically integrate the flux/counts in the line.
   """

   """ Plot the data over this spectral range """
   mask = (wavelength>bluemin) & (wavelength<=redmax)
   tmplamb = wavelength[mask].copy()
   tmpflux = flux[mask].copy()
   if(var):
      tmpvar = var[mask].copy()
      plot_spectrum_array(tmplamb,tmpflux,tmpvar)
   else:
      plot_spectrum_array(tmplamb,tmpflux)

   """ Find a linear fit to the background regions """
   bkgdmask = ((tmplamb>bluemin) & (tmplamb<bluemax)) | \
       ((tmplamb>redmin) & (tmplamb<redmax))
   bkgdwave = tmplamb[bkgdmask].copy()
   bkgdflux = tmpflux[bkgdmask].copy()
   bkgdpoly = numpy.polyfit(bkgdwave,bkgdflux,1)
   continuum = tmplamb*bkgdpoly[0] + bkgdpoly[1]
   plt.plot(tmplamb,continuum,'r')
   plt.axvline(bluemin,color='k')
   plt.axvline(bluemax,color='k')
   plt.axvline(redmin,color='k')
   plt.axvline(redmax,color='k')
   plt.xlim(tmplamb[0],tmplamb[tmplamb.size - 1])

   """ Calculate the subtracted spectrum, and plot it if desired """
   subflux = tmpflux - continuum

   if(showsub):
      plt.figure()
      plt.clf()
      plt.plot(tmplamb,subflux)
      plt.xlim(tmplamb[0],tmplamb[tmplamb.size - 1])

   """ Numerically integrate the flux/counts in the line region """
   linemask = numpy.logical_not(bkgdmask)
   linewave = tmplamb[linemask].copy()
   lineflux = subflux[linemask].copy()
   # Assume that the wavelength scale is linear
   delwave = linewave[1] - linewave[0]
   print delwave
   intflux = (lineflux * delwave).sum()
   print intflux

