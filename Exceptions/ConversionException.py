class ConversionError(Exception):
    def __init__(self, *args, **kwargs):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)

class EmptyPageConversionError(Exception):
    def __init__(self, *args, **kwargs):
        # Call the base class constructor with the parameters it needs
        super().__init__(*args, **kwargs)
