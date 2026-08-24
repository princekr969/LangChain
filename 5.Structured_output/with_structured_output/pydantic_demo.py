from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str='Prince'
    age: Optional[int] = None
    email:EmailStr
    cgpa: float = Field(gt=0, lt=10, default=0, description='A decimal value representing')

new_student = {'age':'22', 'email':'pr@gmail.com', 'cgpa':82}

student = Student(**new_student)
student_dict = dict(student)
student_json = student.model_dump_json()

print(student)