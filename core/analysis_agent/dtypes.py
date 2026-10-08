"""Normalize observed SQL/pandas type labels before choosing an EDA strategy."""
import re

def family(value):
    dtype = str(value or '').strip().casefold()
    # Match complete type names. 'long' must not classify MySQL longtext.
    base = re.split(r'[\s(<\[]', dtype, maxsplit=1)[0]
    if base in {'bool','boolean','object','string','str','category','categorical',
                'char','varchar','nvarchar','nchar','text','tinytext','mediumtext','longtext','enum','set'}:
        return 'categorical'
    if (base in {'float','double','decimal','numeric','number','real','long','short','byte',
                 'tinyint','smallint','mediumint','bigint','integer','int'}
            or re.fullmatch(r'(?:u?int|float|decimal)\d+(?:,\d+)?',base)):
        return 'numeric'
    if base in {'date','datetime','timestamp','datetime64','timedelta64'}:
        return 'temporal'
    return 'other'
