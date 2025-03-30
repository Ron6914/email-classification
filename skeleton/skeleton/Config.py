class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['Type 2', 'Type 3', 'Type 4']  # Your target columns
    CLASS_COL = 'Type 2'  # Primary target column
    GROUPED = 'Type 1'    # Grouping column (if used)
    CLASS_WEIGHTS = 'balanced'  # Options: None, 'balanced', or custom dict


    # Data paths
    DATA_PATH = 'data/'  # Path to your data files
    MODEL_SAVE_PATH = 'models/'  # Where to save trained models

    MISSING_CATEGORY = 'MISSING'  # Default value for NaN in target columns

    # Model parameters
    RANDOM_SEED = 42  # For reproducibility
    TEST_SIZE = 0.2    # Train-test split ratio

    # Text processing
    MAX_FEATURES = 5000  # For TF-IDF vectorizer
    TEXT_COLUMNS = [TICKET_SUMMARY, INTERACTION_CONTENT]  # Columns to combine