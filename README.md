# Multi-agent-bug-fixer
For anyone other than the professor, this repo is not complete!
A .env file is not included, before running add a .env file to the root of the project, it should consist of the API keys to the platforms that models belong to. To run directly, add a together_ai api key. Run the following commands at the project root.  
to set up:   
docker build \   
  -t swe-lite-runner \
  -f docker/Dockerfile \                     
  .

to run:    
docker run --rm \
  --env-file .env \   
  -v "$HOME/.cache/swe-lite/mirror:/mirror" \
  swe-lite-runner <<instance_id from swe-bench-lite>>     

  Note for professor: I tried to use the llm with a high temperature out of curiosity and left it that way, I just realized that. I am not making any changes right now to the project as I do not want to violate the deadline, if you plan on running the project on your local changing the temperature to zero would let you solely observe the bug related to virtual environment setting or the other ones that I was not able to identify and fix before the deadline. Sorry for the inconvenience. 
