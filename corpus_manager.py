import os
import xml.etree.ElementTree as ET
from datetime import datetime
import json


class CorpusManager:

    def __init__(self, name: str, filename: str, from_xml: bool = True):
        """
        The constructor of the class CorpusManager. Loads a query serialized as XML. It is assumed that the document is located in the directory ./data .
        All query attributes are incorporated in a object variable self.corpus (dict).

        Args:
            name: The name of the object/corpus.
            filename: The filename of the xml document.
        """

        self.corpus = {}
        self.name = ""

        if from_xml:
            self.deserialize_corpus_from_xml(name, filename)
        else:
            self.deserialize_corpus_from_json(filename)

    def deserialize_corpus_from_xml(self, name, filename) -> None:

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

        self.name = filename

        with open(os.path.join("data/processed", filename), "r", encoding='utf-8') as f:
            self.corpus = json.load(f)

    def serialize_corpus(self, filename: str) -> None:

        with open(os.path.join("data/processed", filename), "w", encoding='utf-8') as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2, default=CorpusManager.json_converter)

    @staticmethod
    def json_converter(obj):
        if isinstance(obj, datetime):
            return obj.date().isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def filter_by_title(self):
        pass


if __name__ == "__main__":
    corpus = CorpusManager(name="Test", filename="test", from_xml=False)
    print(corpus.corpus)
    corpus.serialize_corpus("test")
