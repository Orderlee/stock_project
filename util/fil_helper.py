from dataclasses import dataclass


@dataclass
class FileReader:

    context: str =''
    fname: str=''
    train: object = None
    test: object = None
    id: str =''
    label: str =''

    def new_file(self) -> str:
        pass

    def csv_to_frame(self) -> object:
        pass

    def jseon_lead(self):
        pass


