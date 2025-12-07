# langie-agent-code

# Files
runpy.py
Code to run the main agent 

common_mpc_server.py
the common server

atlas_mpc_server.py
The atlas server


# to run 
  uvicorn common_mcp_server:app --reload --port 8001


  uvicorn atlas_mcp_server:app --reload --port 8002


  python runpy.py


  With postman requests for human 
  
  link http://localhost:9005/human-review/decision

  
  Json payload 

  
  {"thread_id": "ba4d445e-0e92-47ff-b7fc-feda229b0d3f", "decision": "ACCEPT"}


# Introduction video
https://drive.google.com/file/d/18nM5oUV48-5yYNj9VBlQTmNUgagogD_L/view?usp=sharing

# Code explanation video 
https://drive.google.com/file/d/12MFoWSA_dt0Nn32IMPGPG2wZgAfKJ4c_/view?usp=sharing

video explanation
