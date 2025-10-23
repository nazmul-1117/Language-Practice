from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

@app.get('/')
def root():
    return {
        'message':{
            'data' : 'Hi Nazmul, I am API03-Home',
            'code' : 17
        },
        'success' : True
    }

class model(BaseModel):
    id: int
    title: str
    description: str
    deleted: bool
    image: Optional[str]

@app.post('/blogs')
def create_blogs(request: model):
    return {
        'message': {
            'msg': 'successfull',
            'id': request.id,
            'image' : request.image
        },
        'code': 201
    }

@app.get('/blogs')
def get_blogs():
    return {
        'message': {
            'msg': 'successfull',
            # 'title': model.title
            # 'image' : model.image
        },
        'code': 200
    }