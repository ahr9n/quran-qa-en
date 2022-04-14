mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

pip install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html
