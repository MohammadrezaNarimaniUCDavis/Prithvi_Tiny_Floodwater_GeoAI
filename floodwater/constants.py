"""Sen1Floods11 v1.1 and Prithvi-EO-2.0 spectral constants."""

S2_BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
# Prithvi-EO-2.0 input order: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
PRITHVI_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
PRITHVI_BAND_INDICES = [S2_BAND_NAMES.index(b) for b in PRITHVI_BANDS]
PRITHVI_BAND_LABELS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
I_BLUE, I_GREEN, I_RED, I_NIR, I_SWIR1, I_SWIR2 = range(6)

REFLECTANCE_SCALE = 1e-4  # Level-1C TOA reflectance stored as int16 times 10000

LABEL_NOWATER, LABEL_WATER, LABEL_INVALID = 0, 1, -1
IGNORE_INDEX = -1

# Official HLS pretraining statistics (surface reflectance times 10000)
PRITHVI_PRETRAIN_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_PRETRAIN_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

CHIP_SIZE = 512
CROP_SIZE = 256
