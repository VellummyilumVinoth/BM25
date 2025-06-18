from abc import ABC, abstractmethod
from typing import Union, List, Dict

# Type alias for sparse vector
SparseVector = Dict[str, Union[List[int], List[float]]]


class BaseSparseEncoder(ABC):
    """Abstract base class for sparse encoders"""

    @abstractmethod
    def encode_documents(
            self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        encode documents to a sparse vector (for upsert to pinecone)

        Args:
            texts: a single or list of documents to encode as a string
        """
        pass  # pragma: no cover

    @abstractmethod
    def encode_queries(
            self, texts: Union[str, List[str]]
    ) -> Union[SparseVector, List[SparseVector]]:
        """
        encode queries to a sparse vector

        Args:
            texts: a single or list of queries to encode as a string
        """
        pass  # pragma: no cover