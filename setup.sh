mkdir -p ~/.streamlit/

pip install streamlit
pip uninstall torch
pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# pip install git+https://github.com/deepset-ai/haystack.git

# --find-links https://download.pytorch.org/whl/torch_stable.html
# torch==1.7.0+cpu

echo "\
[general]\n\
email = \"juan.ciro@premexcorp.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
