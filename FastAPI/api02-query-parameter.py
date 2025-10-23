from fastapi import FastAPI
from typing import Optional
from typing import Union

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'Hello from FastAPI! My name is Nazmul | I am from Home'}

@app.get('/about')
def about():
    return {'message': 'Hello from FastAPI! My name is Nazmul | I am from About'}

# http://127.0.0.1:8000/blogs?page=100&limit=500&premium=true&sort=asc
@app.get('/blogs')
def get_blogs(page: int = 1, limit: int = 10, premium: bool = False, sort: Optional[str] = None):
    
    if premium:
        return{
            'data':{
                'page': page,
                'limit': limit,
                'sort': sort,
                'premium': premium
            },
            'status': 'success'
        }
    else:
        return{
            'data':{
                'page': page,
                'limit': limit,
                'sort': sort,
                'premium': 'Not Applicable'
            },
            'status': 'success'
        }

#  Use Union instead of Optional, if you want to accept multiple types [str or none]   
@app.get('/comments/{id}')
def get_comments(id: int, query: Union[str, None] = None):
    return{
        'data':{
            'id': id,
            'query': query
        },
        'status': 'success'
    }


