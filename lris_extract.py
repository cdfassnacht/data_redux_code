"""
lris_extract.py

Program to extract 1D spectra from reduced 2D LRIS spectra.  These reduced
 spectra should be rectified and sky-subtracted, with the dispersion running
 along the x-axis.  The niredux code in the nirspec directory will produce
 such files.
"""

import numpy
import numpy as np
from scipy import interpolate
from mostools import spectools as st
import spec_simple as ss
import ccdredux as c
import pyfits as p
import pylab as plot
from scipy import interpolate,ndimage
from math import sqrt,log

#-----------------------------------------------------------------------

def clear_all(nfig=10):
   """
   Clears all figure windows.
   """

   for ica in range(1,nfig+1):
      plot.figure(ica)
      plot.clf()

#-----------------------------------------------------------------------

def sig_clip(data,sigthresh=3.,range_only=True):
   """
   Standard sigma clipping algorithm. Used specifically to create a better
   range when plotting the output spectrum. If range_only is true, only
   the sigma-clipped range is output.
   """
   while len(data[np.fabs(data-np.median(data)) > np.std(data)*sigthresh]) > 0:
      data = data[np.fabs(data-np.median(data)) <= np.std(data)*sigthresh]
   if range_only:
      return np.min(data),np.max(data)
   else:
      return data

#-----------------------------------------------------------------------

def read_lris_spec(filename, weightfilename=None, weightfiletype='InverseVariance', x1=0, x2=0, y1=0, y2=0, informat='new', verbose=True, trimfile=False, weighted_var=True,crval1=None):
   """
   Reads in a 2D spectrum produced by Matt Auger's lrisedux code.
   New format (default):
      hdu0 = 2D reduced spectrum
      hdu1 = 2D variance spectrum
   Old format:
      hdu0 = 2D reduced spectrum
      hdu1 = 1D wavelength vector
      hdu2 = 1D sky vector

   Inputs:
      filename - name of file containing reduced spectrum
      x1, etc. - numbers defining the section to trim out of the 2D spectrum,
                 if non-zero.  Defaults = 0
      informat - format of input file (see above). Default = 'new'
      verbose  - set to False to eliminate output
   Outputs:
      d - science data
      v - variance
      w - wavelength
   """

   if(verbose):
      print "Reading file %s." % filename

   try:
      hdulist = p.open(filename)
   except:
      hdulist = p.open(filename,ignore_missing_end=True)

   hdulist.info()

   if weightfilename != None:
      try:
         whdulist = p.open(weightfilename)
      except:
         whdulist = p.open(weightfilename,ignore_missing_end=True)

   """ Trim the data if requested """
   if trimfile:
      xt1,xt2,yt1,yt2 = c.define_trimsec(hdulist[0],x1,x2,y1,y2)
      d = hdulist[0].data[yt1:yt2,xt1:xt2].copy()
      if weightfilename != None:
         v = whdulist[0].data[yt1:yt2,xt1:xt2].copy()
   else:
      d = hdulist[0].data.copy()
      xt1,xt2,yt1,yt2 = 0,np.shape(d)[1],0,np.shape(d)[0]
      if weightfilename != None:
         v = whdulist[0].data
   if informat=='old':
      w = hdulist[1].data.copy()
      v = hdulist[2].data.copy()
   else:
      hdr = hdulist[0].header
      if weightfilename == None: 
         if trimfile:
            v = numpy.median(hdulist[1].data[yt1:yt2,xt1:xt2].copy(),axis=0)
         else:
            v = numpy.median(hdulist[1].data.copy(),axis=0)
         wsize = v.size
      if weightfiletype == 'InverseVariance':
         v[v!=0] = 1./v[v!=0]
      elif ((weightfiletype != 'Variance') & (weightfiletype != 'None')):
         sys.exit("Weight file must be of type InverseVariance, Variance, or None")
      if weightfilename != None:
         vt = np.zeros(np.shape(v)[1])
         for inm in range(0,np.shape(v)[1]): vt[inm] = numpy.median(v[:,inm])
         if not weighted_var: v = vt
         wsize = vt.size
      if crval1 == None:
         w = (1.0*numpy.arange(wsize) + 1.0*xt1 - hdr['crpix1'])*hdr['cd1_1'] + hdr['crval1']
      else:
         w = (1.0*numpy.arange(wsize)-hdr['crpix1'])*hdr['cd1_1'] + crval1
   hdulist.close()
   whdulist.close()
   return d,w,v

#-----------------------------------------------------------------------

def lris_extract_(d,w,v,s,gain,rdnoise,outname,informat='new',outformat='text', apmin=-4., apmax=4.,
                muorder=3, sigorder=3, fitrange=None, weight='gauss', 
                owave=None, do_plot=True, do_subplot=True, 
                stop_if_nan=True, weighted_var=True, output_plot = None, output_plot_dir = None,dispaxis='x',nan_to_zero=True,return_data=False):

   bkgd,amp0=None,None

   """ Show the 2d spectrum """
   mp,sp = c.sigma_clip(d)
   if do_plot:
      plot.figure(1)
      plot.clf()
      plot.imshow(d,origin='lower',cmap=plot.cm.gray,vmin=mp-sp,vmax=mp+4*sp)

   """ Find the trace """
   if(do_plot):
      plot.figure(2)
      plot.clf()
   #mu0,sig0,amp0,bkgd = ss.find_trace(d,apmin=apmin,apmax=apmax,do_plot=do_plot,do_subplot=do_subplot,findmultiplepeaks=False,get_bkgd=False,get_amp=False)
   mu0,sig0 = ss.find_trace(d,apmin=apmin,apmax=apmax,do_plot=do_plot,do_subplot=do_subplot,findmultiplepeaks=False,get_bkgd=False,get_amp=False)

   """ Trace the object down the 2d spectrum """
   mupoly,sigpoly = ss.trace_spectrum(d,mu0,sig0,fitrange=fitrange,
                                      muorder=muorder,sigorder=sigorder,
                                      do_plot=do_plot,do_subplot=do_subplot)

   """ Extract the 1d spectrum """
   spec,varspec = ss.extract_spectrum(d,mupoly,sigpoly,apmin=apmin,apmax=apmax,
                                      weight=weight,sky=s,gain=gain,
                                      rdnoise=rdnoise,do_plot=do_plot,
                                      do_subplot=do_subplot,weighted_var=True, var_in=v,bkgd=bkgd,amp0=amp0)
   if informat=='new':
      if not weighted_var: varspec = v
   #calculating varspec frome extract_spectrum is pointless since it is just redefine here

   """ Check for NaN values in the returned spectra """
   spec_nan = False
   
   if nan_to_zero: 
      gnan = np.union1d(np.arange(len(spec))[np.isnan(spec)],np.arange(len(varspec))[np.isnan(varspec)])
      spec[gnan],varspec[gnan] = 0,0
   
   if ((numpy.isnan(spec)).sum() > 0 or (numpy.isnan(varspec)).sum() > 0):
      spec_nan = True

   if (stop_if_nan and spec_nan):
      print ""
      print "WARNING: There is at least one NaN in the extracted spectrum "
      print " or variance spectrum.  Stopping here. "
      print "If you understand the cause of this/these NaN value(s) and want"
      print " to continue, re-run lris_extract with stop_if_nan=False"
      print ""
      return
   elif (spec_nan):
      spec[numpy.isnan(spec)] = 0.0
      varspec[numpy.isnan(varspec)] = 2.0 * numpy.nanmax(varspec)

   """ 
   Linearize the wavelength scale if needed (taken from code by Matt Auger) 
     Old format: explicitly linearize here
     New format: wavelength scale is alread linearized by lrisedux
   """

   if owave is None:
      if informat=='old':
         w0 = w[0]
         w1 = w[-1]
         owave = numpy.linspace(w0,w1,w.size)
      else:
         owave = w
   
   tmpwave,outspec = ss.resample_spec(w,spec,owave)
   tmpwave,outvar  = ss.resample_spec(w,varspec,owave)
   if not weighted_var: tmpwave,sky     = ss.resample_spec(w,s,owave)

   """ Plot the linearized spectrum, with rms """
   rms = numpy.sqrt(outvar)
   if np.shape(spec[0]) == ():
      if do_plot:
         if (do_subplot):
            plot.figure(3)
         else:
         # Re-use fig 4, which showed an earlier version of the extracted 
         # spectrum
            plot.figure(4)
         plot.clf()
         plot.axhline(color='k')
         plot.plot(owave,outspec,linestyle='steps')
         plot.plot(owave,rms,'r',linestyle='steps')
         plot.xlabel('Wavelength')
         plot.ylabel('Relative Flux Density')
         plot.title('Output Spectrum')
         plot.xlim([owave[0],owave[-1]])
         ylbt,yubt = sig_clip(outspec,sigthresh=5.)
         plot.ylim(ylbt,yubt)

         if output_plot != None: 
#         for ip in range(0,4):
#            plot.figure(ip+1)
#            plot.savefig('%s%i.ps'%(output_plot,ip+1))
            plot.figure(2)
            outplotname = 'outplots.%s'%output_plot
            if output_plot_dir != None: outplotname = '%s/%s'%(output_plot_dir,outplotname)
            plot.savefig(outplotname)
            if do_subplot:
               plot.figure(3)
            else:
               plot.figure(4)
            outplotname = 'specplot.%s'%output_plot
            if output_plot_dir != None: outplotname = '%s/%s'%(output_plot_dir,outplotname)
            plot.savefig(outplotname)
   else:
      if do_plot:
         plots_arr = np.array([[3,6,8,10],[2,5,7,9]])
         for idp in range(0,len(spec[0])):
            if (do_subplot):
               plot.figure(plots_arr[0][idp])
               plt.clf()
            else:
         # Re-use fig 4, which showed an earlier version of the extracted 
         # spectrum
               plot.figure(4)
            plot.clf()
            plot.axhline(color='k')
            plot.plot(owave,outspec[:,idp],linestyle='steps')
            plot.plot(owave,rms[:,idp],'r',linestyle='steps')
            plot.xlabel('Wavelength')
            plot.ylabel('Relative Flux Density')
            plot.title('Output Spectrum')
            plot.xlim([owave[0],owave[-1]])
            ylbt,yubt = sig_clip(outspec[:,idp],sigthresh=5.)
            plot.ylim(ylbt,yubt)

            if output_plot != None: 
#         for ip in range(0,4):
#            plot.figure(ip+1)
#            plot.savefig('%s%i.ps'%(output_plot,ip+1))
               plot.figure(plots_arr[1][idp])
               outplotname = 'outplots.%s'%output_plot
               if idp > 0: outplotname = 'outplots_line%i.%s'%(idp+1,output_plot)
               if output_plot_dir != None: outplotname = '%s/%s'%(output_plot_dir,outplotname)
               plot.savefig(outplotname)
               if do_subplot:
                  plot.figure(plots_arr[0][idp])
               else:
                  plot.figure(4)
               outplotname = 'specplot.%s'%output_plot
               if idp > 0: outplotname = 'specplot_line%i.%s'%(idp+1,output_plot)
               if output_plot_dir != None: outplotname = '%s/%s'%(output_plot_dir,outplotname)
               plot.savefig(outplotname)

   if return_data:
      return owave,outspec,outvar

#-----------------------------------------------------------------------


def lris_extract(filename, outname, weightfile=None,trimfile=False,x1=0, x2=0, y1=0, y2=0,
                informat='new',outformat='text', apmin=-4., apmax=4.,
                muorder=3, sigorder=3, fitrange=None, weight='gauss', 
                owave=None, do_plot=True, do_subplot=True, 
                stop_if_nan=True, weighted_var=True, output_plot = None, output_plot_dir = None,crval1=None,dispaxis='x',findmultiplepeaks=False,return_data=True,nan_to_zero=False,maxpeaks=2):
   #clear_all()
   """
   Given the calibrated and sky-subtracted 2d spectrum, extracts the 1d 
    spectrum, taking into account things specific to the form of the 2d 
    LRIS spectra.
   """ 

   """ Set up NIRSPEC detector characteristics """
   gain = 4.     # Value in e-/ADU
   rdnoise = 25. # Value in e-

   """ Read the 2d spectrum """
   print ""
   d,w,v = read_lris_spec(filename,weightfilename=weightfile,x1=x1,x2=x2,y1=y1,y2=y2,trimfile=trimfile,informat=informat,weighted_var=weighted_var,crval1=crval1)
   if informat=='old':
      s = v
   else:
      s = numpy.sqrt(v)
   bounds_arr = np.array([0,np.min(np.shape(d))])
   fitmp,fixmu = False,False
   if findmultiplepeaks:
      if np.min(np.shape(d)) > 3+(1+2*np.fabs(apmin)+2*apmax)*(maxpeaks-1): 
         fitmp,fixmu,mp_out,bounds_arr = ss.find_multiple_peaks(d,maxpeaks=maxpeaks,output_plot=output_plot,output_plot_dir=output_plot_dir)
      else:
         mpflag,ipk = False,maxpeaks-1
         while ((not mpflag) & (ipk > 1)):
            if np.min(np.shape(d)) > 3+(1+2*np.fabs(apmin)+2*apmax)*(ipk-1): 
               fitmp,fixmu,mp_out,bounds_arr = ss.find_multiple_peaks(d,maxpeaks=ipk,output_plot=output_plot,output_plot_dir=output_plot_dir)
               mpflag = True
            ipk -= 1
   if fitmp:
      numpeaks = np.shape(mp_out)[1]
      mutmp = mp_out[1]
      for imle in range(0,numpeaks):
         output_plot_tmp = output_plot
         if output_plot !=  None:
            if imle != 0: output_plot_tmp = 'line%i.'%(imle+1) + output_plot
         dlb,dub = bounds_arr[2*imle],bounds_arr[2*imle+1]
         if dispaxis == 'x':
            owavet,outspect,outvart = lris_extract_(d[dlb:dub+1,:],w,v[dlb:dub+1,:],s[dlb:dub+1,:],gain,rdnoise,outname,informat=informat,outformat=outformat, apmin=apmin, apmax=apmax,muorder=muorder, sigorder=sigorder, fitrange=fitrange, weight=weight,owave=None, do_plot=do_plot, do_subplot=do_subplot, stop_if_nan=stop_if_nan, weighted_var=weighted_var, output_plot = output_plot_tmp, output_plot_dir = output_plot_dir,dispaxis=dispaxis,nan_to_zero=nan_to_zero,return_data=return_data)
         else:
            owavet,outspect,outvart = lris_extract_(d[:,dlb:dub+1],w,v[dlb:dub+1,:],s[:,dlb:dub+1],gain,rdnoise,outname,informat=informat,outformat=outformat, apmin=apmin, apmax=apmax,muorder=muorder, sigorder=sigorder, fitrange=fitrange, weight=weight,owave=None, do_plot=do_plot, do_subplot=do_subplot, stop_if_nan=stop_if_nan, weighted_var=weighted_var, output_plot = output_plot_tmp, output_plot_dir = output_plot_dir,dispaxis=dispaxis,nan_to_zero=nan_to_zero,return_data=return_data)
         if imle == 0:
            owave,outspec,outvar = np.zeros((numpeaks,len(owavet))),np.zeros((numpeaks,len(outspect))),np.zeros((numpeaks,len(outvart)))
         owave[imle],outspec[imle],outvar[imle] = owavet,outspect,outvart
   else:
      dlb,dub = bounds_arr[0],bounds_arr[1]
      if dispaxis == 'x':
         owave,outspec,outvar = lris_extract_(d[dlb:dub+1,:],w,v[dlb:dub+1,:],s[dlb:dub+1,:],gain,rdnoise,outname,informat=informat,outformat=outformat, apmin=apmin, apmax=apmax,muorder=muorder, sigorder=sigorder, fitrange=fitrange, weight=weight,owave=None, do_plot=do_plot, do_subplot=do_subplot, stop_if_nan=stop_if_nan, weighted_var=weighted_var, output_plot = output_plot, output_plot_dir = output_plot_dir,dispaxis=dispaxis,nan_to_zero=nan_to_zero,return_data=return_data)
      else:
         owave,outspec,outvar = lris_extract_(d[:,dlb:dub+1],w,v[:,dlb:dub+1],s[:,dlb:dub+1],gain,rdnoise,outname,informat=informat,outformat=outformat, apmin=apmin, apmax=apmax,muorder=muorder, sigorder=sigorder, fitrange=fitrange, weight=weight,owave=None, do_plot=do_plot, do_subplot=do_subplot, stop_if_nan=stop_if_nan, weighted_var=weighted_var, output_plot = output_plot, output_plot_dir = output_plot_dir,dispaxis=dispaxis,nan_to_zero=nan_to_zero,return_data=return_data)
   """ Write the output spectrum in the requested format """
   if(outformat == 'mwa'):
      st.make_spec(outspec,outvar,owave,outname,clobber=True)
   else:
      ss.save_spectrum(outname,owave,outspec,outvar)
      

#-----------------------------------------------------------------------

def plot_trace(datafile, weightfile=None,trimfile=False,x1=0, x2=0, y1=0, y2=0,informat='new', weighted_var=True,dispaxis='x'):
   d,w,v = read_lris_spec(datafile,weightfilename=weightfile,x1=x1,x2=x2,y1=y1,y2=y2,trimfile=trimfile,informat=informat,weighted_var=weighted_var)
   ss.find_trace(d,dispaxis=dispaxis,do_plot=True,do_subplot=False,nofit=True)
