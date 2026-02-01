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
    """Data-related paths for OC22"""
    # Base data directory
    DATA_DIR = BasePath.PROJECT_ROOT / "oc22_data"
    
    # OC22 S2EF-Total directories
    S2EF_TOTAL_DIR = DATA_DIR / "s2ef_total_train_val_test_lmdbs" / "data" / "oc22" / "s2ef-total"
    
    # Original LMDB directories
    TRAIN_DIR = S2EF_TOTAL_DIR / "train"
    VAL_ID_DIR = S2EF_TOTAL_DIR / "val_id"
    VAL_OOD_DIR = S2EF_TOTAL_DIR / "val_ood"
    TEST_ID_DIR = S2EF_TOTAL_DIR / "test_id"
    TEST_OOD_DIR = S2EF_TOTAL_DIR / "test_ood"
    
    # TiO2 filtered LMDB
    TIO2_DIR = DATA_DIR / "tio2_filtered"
    TIO2_TRAIN_DIR = TIO2_DIR / "train"
    TIO2_VAL_ID_DIR = TIO2_DIR / "val_id"
    TIO2_VAL_OOD_DIR = TIO2_DIR / "val_ood"
    TIO2_TEST_ID_DIR = TIO2_DIR / "test_id"
    TIO2_TEST_OOD_DIR = TIO2_DIR / "test_ood"
    
    # Metadata
    METADATA_DIR = DATA_DIR / "metadata"
    
    # Metadata files
    OC22_MAPPING_FILE = METADATA_DIR / "oc22_metadata.pkl"
    FULL_DATA_MAPPING_FILE = METADATA_DIR / "full_data_mapping.json"
    TIO2_MAPPING_FILE = METADATA_DIR / "tio2_data_mapping.json"
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


class OC22DataPath:
    """OC22 dataset specific paths"""
    
    @staticmethod
    def get_s2ef_total_dir():
        """Get S2EF-Total base directory"""
        return DataPath.S2EF_TOTAL_DIR
    
    @staticmethod
    def get_split_dir(split="train"):
        """
        Get directory for specific split
        """
        split_dirs = {
            "train": DataPath.TRAIN_DIR,
            "val_id": DataPath.VAL_ID_DIR,
            "val_ood": DataPath.VAL_OOD_DIR,
            "test_id": DataPath.TEST_ID_DIR,
            "test_ood": DataPath.TEST_OOD_DIR,
        }
        return split_dirs.get(split, DataPath.TRAIN_DIR)
    
    @staticmethod
    def get_tio2_split_dir(split="train"):
        """
        Get TiO2 filtered directory for specific split
        """
        split_dirs = {
            "train": DataPath.TIO2_TRAIN_DIR,
            "val_id": DataPath.TIO2_VAL_ID_DIR,
            "val_ood": DataPath.TIO2_VAL_OOD_DIR,
            "test_id": DataPath.TIO2_TEST_ID_DIR,
            "test_ood": DataPath.TIO2_TEST_OOD_DIR,
        }
        return split_dirs.get(split, DataPath.TIO2_TRAIN_DIR)
    
    @staticmethod
    def get_all_splits():
        """Get list of all available splits"""
        return ["train", "val_id", "val_ood", "test_id", "test_ood"]


class DownloadURLs:
    """URLs for downloading OC22 data"""
    BASE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data"
    
    OC22_METADATA_URL = f"{BASE_URL}/oc22/oc22_metadata.pkl"
    
    OC22_LMDB_URL = f"{BASE_URL}/oc22/s2ef_total_train_val_test_lmdbs.tar.gz"
    
    @staticmethod
    def get_oc22_lmdb_url():
        """Get download URL for OC22 pre-computed LMDBs"""
        return DownloadURLs.OC22_LMDB_URL