from pydantic import BaseModel, Field          # 1. always this import

class YourName(BaseModel):                     # 2. always (BaseModel)
    field_name: type = Field(description="...")  # 3. repeat per field
    #          ^int/str/bool/list[str]            4. pick from ~4 types