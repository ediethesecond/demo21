import os

def print_structure(startpath, exclude_dirs=None, output_file='structure.txt'):
   if exclude_dirs is None:
       exclude_dirs = ['data', '__pycache__', '.git', '.conda']
   
   with open(output_file, 'w', encoding='utf-8') as f:
       f.write(f"Current directory: {os.getcwd()}\n\n")
       f.write("Directory structure:\n")
       
       for root, dirs, files in os.walk(startpath):
           # Remove excluded directories
           dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.') and d != 'data']
           
           level = root.replace(startpath, '').count(os.sep)
           indent = ' ' * 4 * level
           f.write(f'{indent}{os.path.basename(root)}/\n')
           subindent = ' ' * 4 * (level + 1)
           for file in files:
               f.write(f'{subindent}{file}\n')

# Run from your project root
exclude_dirs = [
   'data', 
   '__pycache__', 
   '.git', 
   '.conda'
]
print_structure('.', exclude_dirs=exclude_dirs)
print("Structure has been written to structure.txt")