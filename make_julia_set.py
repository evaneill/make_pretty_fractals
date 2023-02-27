import os
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import numpy as np
from tqdm import tqdm

BLACK = "#000000"
WHITE = "#ffffff"

GRAYSCALE = {
  0. : WHITE,
  1. : BLACK,
}

PRETTIER_COLORMAP = {
    0.0 : BLACK,
    0.05: "#2a52be", # cerulean blue
    0.1 : "#fada5e", # naples yellow
    1.0 : WHITE,
}

def make_image_array(
  c : complex,
  size : tuple, 
  power = 2, 
  niter = 256,
  _from=None,
  _to=None,
  ) -> np.array:
  """
  Create an np.array of size `size`, wherein each value represents the number of 
  iterations that underlying complex values can go through the iteration  z:= z^k + c
  before satisfying |z|>2.

  Args:
      c (complex): 
        The `c` in the iteration function z:=z^k + c.

      size (tuple): 
        number of pixels (wide, tall).

      power (int, optional): 
        The power in the iteration function z:=z^k+c. 
        Defaults to 2.

      niter (int, optional): 
        The number of iterations to pass the input through, at most. In general, you 
        increasing this only affects the quality of rendering on fine details.
        Defaults to 256.

      _from (complex, optional): 
        The upper leftmost bound of values to use.
        If None, will assume it's 2^(1/power) (-1. + i).
        This zooms into a typically reasonable range.
        Defaults to None.

      _to (complex, optional): 
        The lower rightmost bound of values to use.
        If None, will assume it's 2^(1/power) (1. - i).
        This zooms into a typically reasonable range.
        Defaults to None.

  Returns:
      np.array: 
        An array of integers, each of which represents the number of iterations the 
        corresponding complex value went through until reaching |z|>2. Can be at most 
        `niter`, which means it stayed within the bounds for all iterations.
  """
  if _from is None:
    # nth root of 2
    factor = np.power(2,1/power)
    _from = complex(-factor,factor)

  if _to is None:
    # nth root of 2
    factor = np.power(2,1/power)
    _to = complex(factor,-factor)

  w, l = size

  results = np.zeros((w,l),dtype=np.int16)
  
  x = np.linspace(_from.real,_to.real,w,dtype=np.float32)
  y = np.linspace(_from.imag,_to.imag,l,dtype=np.float32)
  
  z_r,z_i = np.meshgrid(x,y)

  print("Setting up matrix...")
  # z = z_x + i z_y
  z = z_r + complex(0,1)*z_i
  
  # Make sure it's cast as (float32, float32)
  z = z.astype(np.complex64)

  print("Iterating...")
  for _ in tqdm(range(niter)):

    # m = {z | |z|<=2}
    m = (np.abs(z)<=2)
    
    # break early if everything has gone out of bounds
    if (~m).all():
      break

    # Iterate z := z^k + c
    z[m] = np.power(z[m],power) + c
    
    # If it's never been out of bounds, it lasted another iteration
    results[m] += 1

  return results

def main(
  c : complex,
  w : int,
  h : int,
  power=2,
  niter=256,
  colormap=None,
  _from=None,
  _to=None,
  outpath=None,
  filename=None,
  norm_to_maxval=True,
  ) -> None:
  """
  Generate and save an image of an approximation to a julia set, using 
  a sample of complex values in a cartesian grid. Each value in the grid will end up
  corresponding to a pixel in the resulting image.

  Optionally, can make new colormap dictionaries to color the image array as you desire.
  The colormap dictionary just has to be a correspondence from numbers in [0,1] to 
  hex colors.

  Args:
      c (tuple,complex): 
        The `c` in the iteration function z:=z^k + c.
        If specified as a tuple, will assume that the 0th and 1st entry of the tuple 
        are the real and imaginary component, respectively.

      w (int): 
        The number of pixels wide that the image should be.
        (Note that without specifying _from and/or _to, the default will be to 
        sample a cartesian square of complex values. The number of pixels controls
        the number of points sampled in the real/imaginary directions.)

      h (int): 
        The number of pixels tall that the image should be.
        (Note that without specifying _from and/or _to, the default will be to 
        sample a cartesian square of complex values. The number of pixels controls
        the number of points sampled in the real/imaginary directions.)

      power (int, optional): 
        The power in the iteration function z:=z^k+c. 
        Defaults to 2.

      niter (int, optional): 
        The number of iterations to pass the input through, at most. In general, you 
        increasing this only affects the quality of rendering on fine details. 
        Defaults to 256.

      colormap (dict, optional): 
        A dictionary of { float : hex color} pairs, where the minimum key is 0 and the
        maximum key is 1. The color for any values not in the key will be the linearly 
        interpolated RGB color between the closest keys.
        Defaults to None.

      _from (complex, optional): 
        The upper leftmost bound of values to use.
        If None, will assume it's 2^(1/power) (-1. + i).
        This zooms into a typically reasonable range.
        Defaults to None.

      _to (complex, optional): 
        The lower rightmost bound of values to use.
        If None, will assume it's 2^(1/power) (1. - i).
        This zooms into a typically reasonable range.
        Defaults to None.
      
      outpath (str, optional): 
        The folder into which the image should be saved.
        If None, will save to a folder named for the power of the equation used.
        Defaults to None.
      
      filename (str, optional): 
        The name of the image file to write.
        If None, will save to a folder named for the power of the equation used.
        Defaults to None.

      norm_to_maxval (bool, optional):
        Whether the array of integers should be normalized into [0,1] by dividing
        by the maximum value present, or the maximum number of iterations present.
        Defaults to True.
  """
  if isinstance(c,tuple):
    c = complex(c[0],c[1])

  image_array = make_image_array(c, (w,h),power=power,niter = niter,_from=_from,_to=_to)

  if colormap is None:
    colormap = GRAYSCALE

  if isinstance(colormap,dict):
    # Assert ordering on the dictionary (lowest to highest key values)
    colormap = {
      k : colormap[k]
      for k in sorted(colormap.keys())
    }
    colormap = LinearSegmentedColormap.from_list("custom",list(colormap.items()))

  if norm_to_maxval:
    image_array/=image_array.max()
  else:
    image_array/=niter

  im = Image.fromarray(colormap(image_array,bytes=True))

  if outpath is None:
    outpath = f"{power=}_fractals"
  
  if not os.path.exists(outpath):
    os.mkdir(outpath)

  if filename is None:
    filename = f'({round(c.real,3)},{round(c.imag,3)}).png'

  if _from is not None:
    filename = filename.replace(".png",f"{_from=}.png")
  
  if _to is not None:
    filename = filename.replace(".png",f"{_to=}.png")

  full_fpath = os.path.abspath(os.path.join(outpath,filename))
  im.save(full_fpath)

  print(f"Saved the image to {full_fpath}")

if __name__=='__main__':
    import argparse 

    parser = argparse.ArgumentParser()

    parser.add_argument('-c_r',type=float,default = 0,help="Real component of the constant `c` in the iteration z := z^k + c")
    parser.add_argument('-c_i',type=float,default = 0,help="Imaginary component of the constant `c` in the iteration z := z^k + c")

    # total im height, pixels (even only)
    parser.add_argument('-height',type=int,default = 3*(2**9),help="Image height, in pixels (even only).")

    # total im width, pixels (even only)
    parser.add_argument('-width',type=int,default = 3*(2**9),help="Image width, in pixels (even only).")

    # max number iters
    parser.add_argument('-niter',type=int,default=256,help="How many iterations of z:=z^k+c to pass each value through, at most.")
    
    # Power in the iteration equation
    parser.add_argument('-power',type=int,default=2,help="The power `k` in the iteration z := z^k + c.")
    
    # Folder into which images should be saved
    parser.add_argument('-outpath',type=str,default=None,help="The path into which images should be saved.")

    args = parser.parse_args()

    main(
      complex(args.c_r,args.c_i),
      args.height,
      args.width,
      power=args.power,
      niter=args.niter,
      outpath=args.outpath,
      colormap=None,
    )
