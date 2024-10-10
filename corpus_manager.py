import os
import xml.etree.ElementTree as ET
from datetime import datetime
import json


class CorpusManager:
    """
    This class provides methods to load, save and filter serialized query's as xml or json document.
    The documents are saved as object variables in a dictionary that uses the document's title as key. The keys map to
    another dictionary that contains the metadata and the full text of the document.
    The object variable corpus has the following structure:

    {"title": {"source_level": (...),
                "source_name": (...),
                "source_fullname": (...),
                "document_number": (...),
                "document_date": (...),
                "initiator": (...),
                "type": (...),
                "title": (...),
                "url_polx": (...),
                "url": d_element.(...),
                "fulltext": (...)
    }
    }
    """
    def __init__(self, name: str, filename: str, from_xml: bool = True):
        """
        The constructor of the class CorpusManager.

        Args:
            name: The name of the corpus.
            filename: The filename of the xml document.
        """

        self.corpus = {}
        self.name = ""

        if from_xml:
            self.deserialize_corpus_from_xml(name, filename)
        else:
            self.deserialize_corpus_from_json(filename)

    def deserialize_corpus_from_xml(self, name, filename) -> None:
        """
        A helper method for the constructor. Loads a query serialized as XML. It is assumed that the document is located
        in the directory ./data .
        All query attributes are incorporated in the object variable self.corpus (dict).

        Args:
            name: The name of the corpus.
            filename: The filename of the xml document.
        """
        self.corpus = {}
        self.name = name  # e.g. the search word

        xml_file_path = os.path.join("data/", filename)

        try:
            # parse xml document
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")

        # iterate over all document elements
        for d_element in root.findall(".//document"):

            # use title as key
            title = d_element.findtext("title")

            # check if title is already used to avoid collisions
            if title in self.corpus:
                for i in range(2, 100):
                    if f"{title} ({i})" in self.corpus:
                        continue
                    else:
                        title = f"{title} ({i})"
                        break

            # instantiate datetime object
            date_str = d_element.findtext("document_date").strip()
            date = datetime.strptime(date_str, "%Y-%m-%d") if date_str.strip() else ""

            self.corpus[title] = {
                "source_level": d_element.findtext("source_ebene"),
                "source_name": d_element.findtext("source_name"),
                "source_fullname": d_element.findtext("source_fullname"),
                "document_number": d_element.findtext("document_number"),
                "document_date": date,
                "initiator": d_element.findtext("initiator"),
                "type": d_element.findtext("type"),
                "title": title,
                "url_polx": d_element.findtext("document_url_polx"),
                "url": d_element.findtext("document_url"),
                "fulltext": d_element.findtext("fulltext")
            }

    def deserialize_corpus_from_json(self, filename: str) -> None:
        """
        A helper method for the constructor. Loads a serialized CorpusManager object. It is assumed that the object is
        located in the directory ./data/processed .

        Args:
            filename: The filename/name of the serialized corpus.
        """
        self.name = filename

        with open(os.path.join("data/processed", filename), "r", encoding='utf-8') as f:
            self.corpus = json.load(f)

    def serialize_corpus(self, filename: str) -> None:
        """
        This method serializes a corpus.

        Args:
            filename: The filename of the saved object.
        """
        with open(os.path.join("data/processed", filename), "w", encoding='utf-8') as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2, default=CorpusManager.json_converter)

    def filter_by_title(self, keyword: str or list, case_sensitive: bool = False) -> None:
        """
        This method filters an object corpus with a given keyword or a list of keywords. An entry in the corpus is
        deleted if the title does not match the keyword or a keyword in the list, respectively.

        Args:
            keyword: The keyword or the list of keywords.
            case_sensitive: If True, every keyword is treated as case-sensitive.
        """
        i = 0
        keys_to_delete = []

        if isinstance(keyword, str):
            keyword = [keyword]

        for k in self.corpus.keys():

            if not case_sensitive:
                if not any(kw.lower() in k.lower() for kw in keyword):
                    keys_to_delete.append(k)
            else:
                if not any(kw in k for kw in keyword):
                    keys_to_delete.append(k)

        for k in keys_to_delete:
            del self.corpus[k]
            i += 1

        print(f"{i} entries in the corpus were deleted.")

    @staticmethod
    def json_converter(obj) -> str or None:
        """
        A helper method for serialize_corpus, which translates a datetime object to a string.

        Args:
            obj: The object to be checked and transformed if applicable.
        Returns:
            The transformed datetime object as string or None.
        """
        if isinstance(obj, datetime):
            return obj.date().isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
