# Instalação 


 ~~~
 pip install -r requirements.txt
 ~~~


 Para rodar: 

 ~~~
 uvicorn main:app --reload
 ~~~


 ~~~ 
curl -X 'POST' \
  'http://127.0.0.1:8000/prever' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Marca": "Ford",
  "Modelo": "Ka",
  "Ano": 2020,
  "Quilometragem": 145,
  "Cor": "Azul",
  "Cambio": "Manual",
  "Combustivel": "Flex",
  "Portas": 4
}'
 ~~~


Para acessar Swagger:
~~~
http://127.0.0.1:8000/docs
~~~