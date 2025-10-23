from fastapi import FastAPI

app = FastAPI()

# Home Directory
@app.get("/")
def read_root():
    return {"data": "Hello from FastAPI! Basic"}

# Static Routing
@app.get("/about")
def get_about():
    return {
        'data': {
            'name': 'Nazmul',
            'age': 22
        },
        'status': 'success'
    }

# Dynamic Routing String input vs Integer input
# @app.get("/blogs/{author}")
# def get_blogs(author):
#     return{
#         'data':{
#             'author': author,
#             'blog': 'this is a blog for author: ' + author
#         },
#         'status': 'success'
#     }

# Dynamic Routing (int)
@app.get("/blogs/{id}")
def get_blogs(id: int):
    return{
        'data':{
            'id': id,
            'blog': 'this is a blog for id: ' + str(id)
        },
        'status': 'success'
    }

@app.get("/blogs/{id}/comments/")
def get_comments(id: int):
    return{
        'data':{
            'id': id,
            'comment': 'this is a comment for id: ' + str(id)
        },
        'status': 'success'
    }