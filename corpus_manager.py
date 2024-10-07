import os
import xml.etree.ElementTree as ET
from datetime import datetime


class CorpusManager:

    def __init__(self, name: str, filename: str):
        """
        The constructor of the class CorpusManager. Loads a query serialized as XML. It is assumed that the document is located in the directory ./data .
        All query attributes are incorporated in a dictionary self.corpus.

        Args:
            name: The name of the object/corpus.
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
            if title not in self.corpus:

                # instantiate datetime object
                date_str = d_element.findtext("document_date")
                date = datetime.strptime(date_str, "%Y-%m-%d") if date_str.strip() else date_str

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
            else:
                print(title)  # print missing document

    def serialize_corpus(self, path):
        pass

    def filter_by_title(self):
        pass


if __name__ == "__main__":

    corpus = CorpusManager(name="Test", filename="dateninstitut_fulltext.xml")

    print(corpus.corpus)
