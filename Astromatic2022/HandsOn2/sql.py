from urllib.request import urlopen
from urllib.parse import urlencode
import numpy as np


PUBLIC_URL = 'http://cas.sdss.org/public/en/tools/search/sql.aspx'
DEFAULT_FMT = 'csv'
GAL_COLORS_NAMES = ['u', 'g', 'r', 'i', 'z', 'specClass',
                    'redshift', 'redshift_err']
                    
def remove_sql_comments(sql):
    """Strip SQL comments starting with --"""
    return ' \n'.join(map(lambda x: x.split('--')[0], sql.split('\n')))

sql_str = ('\n'.join(("SELECT TOP %i" % 50000,
                             "  p.u, p.g, p.r, p.i, p.z, s.class, s.z, s.zerr",
                             "FROM PhotoObj AS p",
                             "  JOIN SpecObj AS s ON s.bestobjid = p.objid",
                             "WHERE ",
                             "  p.u BETWEEN 0 AND 19.6",
                             "  AND p.g BETWEEN 0 AND 20",
                             "  AND s.class <> 'UNKNOWN'",
                             "  AND s.class <> 'STAR'",
                             "  AND s.class <> 'SKY'",
                             "  AND s.class <> 'STAR_LATE'")))
sql_str = remove_sql_comments(sql_str)
params = urlencode(dict(cmd=sql_str, format=DEFAULT_FMT))
output = urlopen(PUBLIC_URL + '?%s' % params)
kwargs = {'delimiter': ',', 'skip_header': 2,
                  'names': GAL_COLORS_NAMES, 'dtype': None,
                  'encoding': 'ascii',
                  }

data = np.genfromtxt(output, **kwargs)
print(data)
np.save(archive_file, data)
