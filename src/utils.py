import torch 

def detect_environment():
    """ Detects the environment and returns the device to be used 
    Args:
        None
    Returns:
        str: device to be used"""
    if torch.cuda.is_available():
        return 'cuda'
    
    # Check for macOS
    elif torch.backends.mps.is_available():
        return 'mps'
    
    return 'cpu'



if __name__ == '__main__':
    print(detect_environment())