window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})

app.location$.subscribe(function() {
  // added for instant loading of math content without having to refresh page
  // because this mathjax.js is automatically placed at the bottom of html
  // (after successful build, the math rendering is executed last as default)
})
