"""Code reproduced from [1] for compatibility with JAX.

MIT License

Copyright (c) 2019 Todd Karin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[1] https://github.com/toddkarin/global-land-mask

"""
import numpy as onp
import jax.numpy as jnp

_mask_fid = onp.load('globe_combined_mask_compressed.npz')

_mask = jnp.asarray(_mask_fid['mask'])
_lat = jnp.asarray(_mask_fid['lat'])
_lon = jnp.asarray(_mask_fid['lon'])

def lat_to_index(lat):
    """
    Convert latitude to index on the mask

    Parameters
    ----------
    lat : numeric
        Latitude to get in degrees

    Returns
    -------
    index : numeric
        index of the latitude axis.

    """
    lat = jnp.asarray(lat)
    lat = jnp.where(lat > _lat.max(), _lat.max(), lat)
    lat = jnp.where(lat < _lat.min(), _lat.min(), lat)

    return ((lat - _lat[0])/(_lat[1]-_lat[0])).astype(jnp.int32)

def lon_to_index(lon):
    """
    Convert longitude to index on the mask

    Parameters
    ----------
    lon : numeric
        Longitude to get in degrees

    Returns
    -------
    index : numeric
        index of the longitude axis.

    """
    lon = jnp.asarray(lon)
    lon = jnp.where(lon > _lon.max(), _lon.max(), lon)
    lon = jnp.where(lon < _lon.min(), _lon.min(), lon)

    return ((lon - _lon[0]) / (_lon[1] - _lon[0])).astype(jnp.int32)


def is_land(lat, lon):
    """

    Return boolean array of whether the coordinates are on the land. Note
    that most lakes are considered on land.

    Parameters
    ----------
    lat
    lon

    Returns
    -------

    """
    lat_i = lat_to_index(lat)
    lon_i = lon_to_index(lon)

    return jnp.logical_not(_mask[lat_i,lon_i])

if __name__ == '__main__':
    pass
