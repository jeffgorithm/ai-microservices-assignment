from pydantic import BaseModel

class Request(BaseModel):
    bus_stop_code: int
    day: int
    time: int