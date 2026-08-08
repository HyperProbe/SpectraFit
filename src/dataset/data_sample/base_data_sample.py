"""
Base data sample classes providing common functionality for all dataset samples.
"""

import typing
from dataclasses import dataclass


@dataclass
class BaseDataSample:
    """
    Base dataclass providing common properties and dictionary-like access for all dataset samples.
    
    This base class implements the common subscribable functionality that allows both
    attribute-style access (sample.id) and dictionary-style access (sample['id']) for
    backward compatibility with existing code.
    
    Attributes
    ----------
    id : str
        Sample ID (format varies by dataset type).
    patient_id : str
        Patient ID.
    fov : str
        Field of view identifier.
    """
    
    id: str
    patient_id: str
    fov: str
    
    def __getitem__(self, key: str) -> typing.Any:
        """
        Dictionary-style access for backward compatibility.
        
        Parameters
        ----------
        key : str
            The attribute name to access.
            
        Returns
        -------
        Any
            The value of the requested attribute.
            
        Raises
        ------
        KeyError
            If the key does not exist as an attribute.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __setitem__(self, key: str, value: typing.Any) -> None:
        """
        Dictionary-style assignment for backward compatibility.
        
        Parameters
        ----------
        key : str
            The attribute name to set.
        value : Any
            The value to assign to the attribute.
            
        Raises
        ------
        KeyError
            If the key does not exist as an attribute.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists as an attribute.
        
        Parameters
        ----------
        key : str
            The attribute name to check.
            
        Returns
        -------
        bool
            True if the attribute exists, False otherwise.
        """
        return hasattr(self, key)
    
    def keys(self) -> typing.List[str]:
        """
        Get all attribute names, similar to dict.keys().
        
        Returns
        -------
        List[str]
            List of all attribute names.
        """
        return [field.name for field in self.__dataclass_fields__.values()]
    
    def get(self, key: str, default: typing.Any = None) -> typing.Any:
        """
        Get an attribute value with a default if not found.
        
        Parameters
        ----------
        key : str
            The attribute name to access.
        default : Any, optional
            Default value to return if key not found.
            
        Returns
        -------
        Any
            The attribute value or default.
        """
        try:
            return self[key]
        except KeyError:
            return default
