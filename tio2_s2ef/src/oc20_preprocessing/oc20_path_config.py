from pathlib import Path


class BasePath:
    """Base directory paths"""
    # Project root (where README.md, environment.yml)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # Main directories
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    TESTS_DIR = PROJECT_ROOT / "tests"
    DOCS_DIR = PROJECT_ROOT / "docs"


class DataPath:
    """Data-related paths"""
    # Base data directory
    DATA_DIR = BasePath.PROJECT_ROOT / "oc20_data"
    
    # Raw data directories (compressed .tar files and extracted compressed .xz files)
    RAW_DIR = DATA_DIR / "raw"
    RAW_200K_DIR = RAW_DIR / "200k"
    RAW_2M_DIR = RAW_DIR / "2M"
    RAW_20M_DIR = RAW_DIR / "20M"
    RAW_ALL_DIR = RAW_DIR / "all"
    
    # Uncompressed data directories (uncompressed .txt and .extxyz files)
    UNCOMPRESSED_DIR = DATA_DIR / "uncompressed_data"
    UNCOMPRESSED_200K_DIR = UNCOMPRESSED_DIR / "200k"
    UNCOMPRESSED_2M_DIR = UNCOMPRESSED_DIR / "2M"
    UNCOMPRESSED_20M_DIR = UNCOMPRESSED_DIR / "20M"
    UNCOMPRESSED_ALL_DIR = UNCOMPRESSED_DIR / "all"
    
    # TiO2 LMDB directories (filtered TiO2-only LMDB databases)
    TIO2_LMDB_DIR = DATA_DIR / "tio2_lmdb"
    TIO2_LMDB_200K_DIR = TIO2_LMDB_DIR / "200k"
    TIO2_LMDB_2M_DIR = TIO2_LMDB_DIR / "2M"
    TIO2_LMDB_20M_DIR = TIO2_LMDB_DIR / "20M"
    TIO2_LMDB_ALL_DIR = TIO2_LMDB_DIR / "all"
    
    # Metadata directory
    METADATA_DIR = DATA_DIR / "metadata"
    
    # Specific metadata files
    OC20_MAPPING_FILE = METADATA_DIR / "oc20_data_mapping.pkl"
    TIO2_MAPPING_FILE = METADATA_DIR / "tio2_data_mapping.pkl"
    TIO2_SYSTEM_IDS_FILE = METADATA_DIR / "tio2_system_ids.txt"
    TIO2_STATISTICS_FILE = METADATA_DIR / "tio2_statistics.json"
    TIO2_NORMALIZATION_FILE = METADATA_DIR / "tio2_normalization_stats.json"


class OutputPath:
    """Output paths for models, results, logs"""
    # Model checkpoints
    MODELS_DIR = BasePath.PROJECT_ROOT / "models"
    CHECKPOINTS_DIR = BasePath.PROJECT_ROOT / "checkpoints"
    
    # Results and logs
    RESULTS_DIR = BasePath.PROJECT_ROOT / "results"
    LOGS_DIR = BasePath.PROJECT_ROOT / "logs"


class S2EFDataPath:
    """S2EF dataset specific paths"""
    
    @staticmethod
    def get_raw_dir(split="200k"):
        """
        Get raw directory for specific split (contains compressed .tar and .xz files)
        """
        split_dirs = {
            "200k": DataPath.RAW_200K_DIR,
            "2M": DataPath.RAW_2M_DIR,
            "20M": DataPath.RAW_20M_DIR,
            "all": DataPath.RAW_ALL_DIR
        }
        return split_dirs.get(split, DataPath.RAW_200K_DIR)
    
    @staticmethod
    def get_uncompressed_dir(split="200k"):
        """
        Get uncompressed directory for specific split (contains .txt and .extxyz files)
        """
        split_dirs = {
            "200k": DataPath.UNCOMPRESSED_200K_DIR,
            "2M": DataPath.UNCOMPRESSED_2M_DIR,
            "20M": DataPath.UNCOMPRESSED_20M_DIR,
            "all": DataPath.UNCOMPRESSED_ALL_DIR
        }
        return split_dirs.get(split, DataPath.UNCOMPRESSED_200K_DIR)
    
    @staticmethod
    def get_tio2_lmdb_dir(split="200k"):
        """
        Get TiO2 LMDB directory for specific split
        """
        split_dirs = {
            "200k": DataPath.TIO2_LMDB_200K_DIR,
            "2M": DataPath.TIO2_LMDB_2M_DIR,
            "20M": DataPath.TIO2_LMDB_20M_DIR,
            "all": DataPath.TIO2_LMDB_ALL_DIR
        }
        return split_dirs.get(split, DataPath.TIO2_LMDB_200K_DIR)
    
    @staticmethod
    def get_tar_file(split="200k"):
        """
        Get tar file path for download
        """
        filename = f"s2ef_train_{split.upper()}.tar"
        return S2EFDataPath.get_raw_dir(split) / filename
    
    @staticmethod
    def get_lmdb_file(split="200k"):
        """
        Get LMDB file path for TiO2 data
        """
        return S2EFDataPath.get_tio2_lmdb_dir(split) / "data.lmdb"


class DownloadURLs:
    """URLs for downloading data"""
    BASE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data"
    
    # OC20 metadata
    OC20_METADATA_URL = f"{BASE_URL}/oc20_data_mapping.pkl"
    
    # S2EF dataset URLs
    S2EF_URLS = {
        "200k": f"{BASE_URL}/s2ef_train_200K.tar",
        "2M": f"{BASE_URL}/s2ef_train_2M.tar",
        "20M": f"{BASE_URL}/s2ef_train_20M.tar",
        "all": f"{BASE_URL}/s2ef_train_all.tar"
    }
    
    @staticmethod
    def get_s2ef_url(split="200k"):
        """Get download URL for S2EF data"""
        return DownloadURLs.S2EF_URLS.get(split, DownloadURLs.S2EF_URLS["200k"])