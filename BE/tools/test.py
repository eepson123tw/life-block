from smolagents import CodeAgent, HfApiModel, tool
from typing import List

@tool
def greeting(query: str) -> str:
    """
    This tool returns a happy greeting to the user.
    
    Args:
        query: The user's name.
    """
    # Example list of catering services and their ratings
    return f"Hello, {query}! How can I help you today?"







def example(**kwargs):
    for key,value in kwargs.items():
        print(f'{key}={value}')

example(a=1,b=2,c=3)


def create_profile(name,age,email):
    print(f'name:{name}')
    print(f'age:{age}')
    print(f'email:{email}')


option = {
    'name': 'tony',
    'age':18,
    'email':'eeeee'
}


create_profile(**option)


# __add__ dunder methods
print((x:=1).__add__(2))
print('a'.__add__('b'))


class ShoppingCart:
    def __init__(self,items:List[str]):
        self.items = items
    def __add__(self,another_cart):
        new_cart = ShoppingCart(self.items+another_cart.items)
        return new_cart
    def __str__(self):
        return f'Cart({self.items})'
    
    def __len__(self):
        return len(self.items)
    
    def __call__(self, *args):
        for arg in args:
            self.items.append(arg)


cart1 = ShoppingCart(['1','2'])
cart2 = ShoppingCart(['3','4'])
cart2('aaaa','gggg')
c3 = cart1 + cart2
print(len(c3),c3)



class add:
    def __init__(self,num:int):
        self.nums = num
    def __str__(self):
        return f'{self.nums}'
    def __add__(self,other): #  addTwo.__add__(5)
        return self.nums + other
    def __call__(self,num):
        return self.nums + num

addTwo = add(2)
addFive = addTwo + 5
print(addFive)#7
print(addTwo(3))#5

import time


def decorator(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        print(f'{func.__name__} is running')
        result = func(*args,**kwargs)
        end_time = time.time()
        print(f'ex time is {start_time - end_time}')
        return result
    return wrapper


@decorator
def square(x):
    return x*x

def print_fn(f,x):
    print(f'{f.__name__} is !!')
    return f(x)

res = print_fn(square,2)

print(res)

