from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = None
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, le=5, default=None)

student_1 = {"age":"32", "email":"abc1@gmail.com", "cgpa":5}

student = Student(**student_1)

print(student)
print(student.age)

s_dict = dict(student)
print(s_dict)
print(s_dict['age'])

s_json = student.model_dump_json()
print(s_json)