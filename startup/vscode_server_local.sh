commit_id=e54c774e0add60467559eb0d1e229c6452cf8447
curl -sSL "https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable" -o vscode-server-linux-x64.tar.gz
mkdir -p ~/.vscode-server/bin/${commit_id}
tar zxvf vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${commit_id} --strip 1


