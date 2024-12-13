# ---Buiid Env
python -m venv .venv 

.venv\Scripts\Activate # (PS) activate env

deactivate #exit Env.

# ---假設要將qat合併到main，確認當前qat已經是最新提交
git checkout main
git merge qat

#---Push GIT
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/MaxLu2002/Resume_Chia-Yang-Lu.git

git push -u origin main # 可以加 --force 覆蓋遠端
git push origin A:B # 遠端分支 B 會被創建或更新，內容來自本地分支A。


#---GIT Branch指令
git branch <new branch>
git switch <target branch>
git -d <branch name> (d= delete , must leave the branch)
git branch -vv 查看當前本地分支關聯到哪個遠端分支


#---創建或導航到目標目錄
mkdir -p ~/my_project
cd ~/my_project