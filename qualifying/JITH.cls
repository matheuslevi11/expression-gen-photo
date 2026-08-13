%%~~~~~~~~~~~~~~~~~~~~~~~
%%~~~~~~~~~~~~~~~~~~~~~~~
%%Padrao JITH 
%%~~~~~~~~~~~~~~~~~~~~~~~
%%~~~~~~~~~~~~~~~~~~~~~~~
%%(1) Inclua os detalhes em "Identificacao.tex".
%%(2) Escreva os capitulos nas pastas apropriadas.
%%(3) Chame os arquivos em "main.tex"
%%(4) Adicione quantas pastas for necessario.
%%~~~~~~~~~~~~~~~~~~~~~~~
%%~~~~~~~~~~~~~~~~~~~~~~~
\NeedsTeXFormat{LaTeX2e} %[2016/04/20]
\ProvidesClass{JITH}[2019/10/24 v4.5  JITH LaTeX document class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{12pt}
\ProcessOptions
\LoadClass{report}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% OTHER PACKAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{fancyhdr}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{setspace}
\RequirePackage{xifthen}
\RequirePackage[svgnames,dvipsnames]{xcolor}
% Idioma: o documento e escrito em INGLES. Para voltar ao portugues, troque
% [english] por [brazilian] AQUI e reverta as strings traduzidas abaixo
% (\capa, \folhaDeRosto, \folhaDeAprovacao). Ver qualifying/NOTES.md.
\RequirePackage[english]{babel}
\RequirePackage[top=3cm,bottom=2cm,right=2cm,left=3cm]{geometry}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% TITLE PAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nomes institucionais mantidos em portugues (nome oficial da instituicao).
% Alternativa em ingles, se o orientador preferir:
%   \def\@university{Federal University of Alagoas}
%   \def\@department{Institute of Computing}
\def\@university{Universidade Federal de Alagoas}
\def\docTitle#1{\gdef\@docTitle{#1}}
\def\author#1{\gdef\@author{#1}}
\def\@department{Instituto de Computação}
\def\docType#1{\gdef\@docType{#1}}
\def\orientador#1{\gdef\@orientador{#1}}
\def\tituloOrientador#1{\gdef\@tituloOrientador{#1}}
\def\coorientador#1{\gdef\@coorientador{#1}}
\def\tituloCoorientador#1{\gdef\@tituloCoorientador{#1}}
%
\def\memberA#1{\gdef\@memberA{#1}}
\def\memberB#1{\gdef\@memberB{#1}}
\def\memberC#1{\gdef\@memberC{#1}}
\def\memberD#1{\gdef\@memberD{#1}}
\def\memberE#1{\gdef\@memberE{#1}}
\def\filiationA#1{\gdef\@filiationA{#1}}
\def\filiationB#1{\gdef\@filiationB{#1}}
\def\filiationC#1{\gdef\@filiationC{#1}}
\def\filiationD#1{\gdef\@filiationD{#1}}
\def\filiationE#1{\gdef\@filiationE{#1}}
%
\def\setDate#1{\gdef\@setDate{#1}}
\def\setMonth#1{\gdef\@setMonth{#1}}
\def\setYear#1{\gdef\@setYear{#1}}
\def\setLocation#1{\gdef\@setLocation{#1}}

\newcommand{\myifnotempty}[2]{\ifthenelse{\not\equal{#1}{}}{#2}{}}

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\capa{%
\pagenumbering{roman}
	\thispagestyle{empty}
	
	\begin{minipage}{1\textwidth}
		\raggedleft \includegraphics[height=35mm]{Cap00/IC.jpg}
	\end{minipage}

	\vfill

    \hspace{-1.5cm}\begin{minipage}{1\textwidth}
		\raggedright \large\textit{{
		    \ifthenelse{\@docType=1}{Undergraduate Final Project}{
		    \ifthenelse{\@docType=2}{Master's Thesis Proposal}{
		    \ifthenelse{\@docType=3}{Master's Thesis}{
		    \ifthenelse{\@docType=4}{Doctoral Qualifying Exam}{
		    \ifthenelse{\@docType=5}{Doctoral Thesis}{
		    E R R O R}}}}}}}
	\end{minipage}
	
	\vspace{5mm}
	
    \hspace{-1.5cm}\scalebox{1.8}{\begin{minipage}{.5\textwidth}
    \renewcommand{\baselinestretch}{0.8}\raggedright\Huge{\textbf\@docTitle}\\
    \end{minipage}}
    
    \vspace{1cm}

    \hspace{-1.0cm}\scalebox{1}{\begin{minipage}{1\textwidth}
		\renewcommand{\baselinestretch}{1}\raggedright {\textbf{\large\@author}}
	\end{minipage}}

    \vspace{1cm}

    \hspace{-1.0cm}\scalebox{1}{\begin{minipage}{1\textwidth}
		\renewcommand{\baselinestretch}{1}\raggedright {\textbf{advised by}}\\
		\large {\textbf{\@tituloOrientador~\@orientador}}
		%\myifnotempty{\@coorientador}{\\ \textbf{\@tituloCoorientador~\@coorientador}}
	\end{minipage}}
	
	\vspace{1cm}
	
	\definecolor{icbackcolor}{RGB}{ 73, 87, 150}
	\definecolor{icfrontcolor}{RGB}{203, 208, 227}
\hfill\fcolorbox{icfrontcolor}{icfrontcolor}{\begin{minipage}[r][2.5cm]{0.6\textwidth}
				\renewcommand{\baselinestretch}{1}\raggedleft\large\@university\\
				\@department\\
				\@setLocation\\
				\@setMonth~\@setDate,~\@setYear
			\end{minipage}}\fcolorbox{icbackcolor}{icbackcolor}{\begin{minipage}[r][2.5cm]{8mm}   \hspace{8mm}\end{minipage}}\hspace{-2cm}
	
	\thispagestyle{empty}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
	
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\renewcommand{\baselinestretch}{1.5} 
\def\folhaDeRosto{%
	\thispagestyle{empty}
	
	\begin{center}
\thispagestyle{empty}
{\MakeUppercase\@university}\\
{\@department}

\vfill	

{\textbf{\large\MakeUppercase\@docTitle}}

\vspace{2cm}

\hspace{.45\textwidth}
\begin{minipage}{.5\textwidth}
	{\ifthenelse{\@docType=1}{Undergraduate Final Project}{
	 \ifthenelse{\@docType=2}{Master's Thesis Proposal}{
	 \ifthenelse{\@docType=3}{Master's Thesis}{
	 \ifthenelse{\@docType=4}{Doctoral Qualifying Exam}{
	 \ifthenelse{\@docType=5}{Doctoral Thesis}{
	 E R R O R}}}}}}
	submitted to the {\@department} of the {\@university} as a partial requirement for the degree of {\ifthenelse{\@docType=1}{Bachelor of Computer Science}{%
	 \ifthenelse{\@docType=2}{Master of Science in Informatics}{%
	 \ifthenelse{\@docType=3}{Master of Science in Informatics}{%
	 \ifthenelse{\@docType=4}{Doctor of Science in Informatics}{%
	 \ifthenelse{\@docType=5}{Doctor of Science in Informatics}{%
	 E R R O R}}}}}}.
\end{minipage}%

\vspace*{\fill}

{\large\@author}\vspace{1cm}

{\textit{\textbf{Advisor: \@tituloOrientador~\@orientador}}}

%\myifnotempty{\@coorientador}{\textit{\textbf{Co-advisor: \@tituloCoorientador~\@coorientador}}}

\vspace{1cm}

\textbf{Examination Committee:}

\vspace{5mm}
\begin{tabular}{lr}
\myifnotempty{\@memberA}{\@memberA&\@filiationA\\}
\myifnotempty{\@memberB}{\@memberB&\@filiationB\\}
\myifnotempty{\@memberC}{\@memberC&\@filiationC\\}
\myifnotempty{\@memberD}{\@memberD&\@filiationD\\}
\myifnotempty{\@memberE}{\@memberE&\@filiationE}
\end{tabular}
\vfill

{\large\@setLocation}

{\large \@setMonth~\@setDate,~\@setYear }

\end{center}
	
	\thispagestyle{empty}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\folhaDeAprovacao{%
	\thispagestyle{empty}

	\begin{center}
	{\MakeUppercase\@university}\\
	{\@department}
	
	\vfill	
	
	{\bfseries\large\MakeUppercase\@docTitle}
	
	\vspace{1cm}
	
	\vspace{5mm}
	
	%\hspace{.45\textwidth}
	\begin{minipage}{.8\textwidth}
		{\ifthenelse{\@docType=1}{Undergraduate Final Project}{
	 \ifthenelse{\@docType=2}{Master's Thesis Proposal}{
	 \ifthenelse{\@docType=3}{Master's Thesis}{
	 \ifthenelse{\@docType=4}{Doctoral Qualifying Exam}{
	 \ifthenelse{\@docType=5}{Doctoral Thesis}{
	 E R R O R}}}}}}
	submitted to the {\@department} of the {\@university} as a partial requirement for the degree of {\ifthenelse{\@docType=1}{Bachelor of Computer Science}{%
	 \ifthenelse{\@docType=2}{Master of Science in Informatics}{%
	 \ifthenelse{\@docType=3}{Master of Science in Informatics}{%
	 \ifthenelse{\@docType=4}{Doctor of Science in Informatics}{%
	 \ifthenelse{\@docType=5}{Doctor of Science in Informatics}{%
	 E R R O R}}}}}}.
	\end{minipage}%
	\vspace*{\fill}


	Approved on \@setMonth~\@setDate,~\@setYear:
	
	\vspace{7mm}\vfill
	
%	\begin{minipage}{.8\textwidth}
%		\begin{center}
\myifnotempty{\@orientador}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@orientador,\\ \@tituloOrientador, Advisor
	\end{tabular}\vspace{7mm}\vfill
}

\myifnotempty{\@coorientador}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@coorientador,\\ \@tituloCoorientador, Co-advisor
	\end{tabular}\vspace{7mm}\vfill
}

\myifnotempty{\@memberA}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@memberA,\\\@filiationA
	\end{tabular}\vspace{7mm}\vfill
}

\myifnotempty{\@memberB}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@memberB,\\\@filiationB
	\end{tabular}\vspace{7mm}\vfill
}

\myifnotempty{\@memberC}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@memberC,\\\@filiationC
	\end{tabular}\vspace{7mm}\vfill
}

\myifnotempty{\@memberD}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@memberD,\\\@filiationD
	\end{tabular}\vspace{7mm}\vfill
}

\myifnotempty{\@memberE}{
	\begin{tabular}{p{8cm}}
		\hline \centering\@memberE,\\\@filiationE
	\end{tabular}\vfill
}
\end{center}

	\thispagestyle{empty}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\dedicatoria{%
    \thispagestyle{empty}
	\input{Cap00/Dedicatoria}
	\thispagestyle{empty}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\agradecimentos{%
    \thispagestyle{empty} 
	\input{Cap00/Agradecimentos}
	\thispagestyle{empty}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\epigrafe{%
    \thispagestyle{empty} 
	\input{Cap00/Epigrafe}
	\thispagestyle{empty}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\abstract{%
	\input{Cap00/Abstract}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\resumo{%
	\input{Cap00/Resumo}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\simbolos{%
	\input{Cap00/Simbolos}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
\def\abreviaturas{%
	\input{Cap00/Abreviaturas}
	\clearpage{\thispagestyle{empty}\cleardoublepage}
}
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% The End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\endinput
%%
%% End of file `JITH.cls'.
