import {
  RuleId,
  fileError,
  fileWarn,
  getMetadataTags,
  remove,
  select,
  selectAll,
  toText,
  writeTexLabelledComment
} from "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-to-tex/dist/types.js
var DEFAULT_IMAGE_WIDTH = 0.7;
var DEFAULT_PAGE_WIDTH_PIXELS = 800;

// ../../node_modules/myst-to-tex/dist/utils.js
function cleanWhitespaceChars(text, nbsp = " ") {
  return text.replace(/\u00A0/g, nbsp).replace(/[\u200B-\u200D\uFEFF]/g, "");
}
var BACKSLASH_SPACE = "\u{1F4A5}\u{1F3AF}BACKSLASHSPACE\u{1F3AF}\u{1F4A5}";
var BACKSLASH = "\u{1F4A5}\u{1F3AF}BACKSLASH\u{1F3AF}\u{1F4A5}";
var TILDE = "\u{1F4A5}\u{1F3AF}TILDE\u{1F3AF}\u{1F4A5}";
var hrefOnlyReplacements = {
  // Not allowed characters
  // Latex escaped characters are: \ & % $ # _ { } ~ ^
  "&": "\\&",
  "%": "\\%",
  $: "\\$",
  "#": "\\#",
  _: "\\_",
  "{": "\\{",
  "}": "\\}",
  "^": "\\^"
};
var textOnlyReplacements = {
  ...hrefOnlyReplacements,
  // quotes
  "\u2019": "'",
  "\u2018": "`",
  "\u201D": "''",
  "\u201C": "``",
  // guillemots
  "\xBB": ">>",
  // These could be improved
  "\xAB": "<<",
  "\u2026": "\\dots",
  "\u2013": "--",
  "\u2014": "---",
  // Commands gobble fhttps://texfaq.org/FAQ-xspaceollowing space
  // See: https://texfaq.org/FAQ-xspace
  "\xA9": "\\textcopyright ",
  "\xAE": "\\textregistered ",
  "\u2122": "\\texttrademark ",
  "<": "\\textless ",
  ">": "\\textgreater ",
  "\xA0": "~",
  "\u202F": "~",
  "\u2009": "\\,"
};
var arrows = {
  "\u2194": "\\leftrightarrow",
  "\u21D4": "\\Leftrightarrow",
  "\u2192": "\\rightarrow",
  "\u21D2": "\\Rightarrow",
  "\u2190": "\\leftarrow",
  "\u21D0": "\\Leftarrow"
};
var symbols = {
  "\u2212": "-",
  // minus
  "-": "-",
  // hyphen minus
  "\uFE63": "-",
  // Small hyphen minus
  "\uFF0D": "-",
  // Full-width Hyphen-minus
  "\uFF0B": "+"
  // Full-width Plus
};
var scripts = {
  "\u2080": "\\textsubscript{0}",
  "\u2081": "\\textsubscript{1}",
  "\u2082": "\\textsubscript{2}",
  "\u2083": "\\textsubscript{3}",
  "\u2084": "\\textsubscript{4}",
  "\u2085": "\\textsubscript{5}",
  "\u2086": "\\textsubscript{6}",
  "\u2087": "\\textsubscript{7}",
  "\u2088": "\\textsubscript{8}",
  "\u2089": "\\textsubscript{9}",
  "\u208A": "\\textsubscript{+}",
  "\u208B": "\\textsubscript{-}",
  "\u208C": "\\textsubscript{=}",
  "\u208D": "\\textsubscript{(}",
  "\u208E": "\\textsubscript{)}",
  "\u2099": "\\textsubscript{n}",
  "\u2070": "\\textsuperscript{0}",
  "\xB9": "\\textsuperscript{1}",
  "\xB2": "\\textsuperscript{2}",
  "\xB3": "\\textsuperscript{3}",
  "\u2074": "\\textsuperscript{4}",
  "\u2075": "\\textsuperscript{5}",
  "\u2076": "\\textsuperscript{6}",
  "\u2077": "\\textsuperscript{7}",
  "\u2078": "\\textsuperscript{8}",
  "\u2079": "\\textsuperscript{9}",
  "\u22C5": "\\textsuperscript{.}",
  "\u207A": "\\textsuperscript{.}",
  "\u207B": "\\textsuperscript{-}",
  "\u207C": "\\textsuperscript{=}",
  "\u207D": "\\textsuperscript{(}",
  "\u207E": "\\textsuperscript{)}",
  \u207F: "\\textsuperscript{n}",
  "\u2071": "\\textsuperscript{i}"
};
var textReplacements = {
  ...textOnlyReplacements,
  ...arrows,
  ...symbols,
  ...scripts
};
var mathReplacements = {
  ...arrows,
  ...symbols,
  "\xBD": "\\frac{1}{2}",
  "\u2153": "\\frac{1}{3}",
  "\u2154": "\\frac{2}{3}",
  "\xBC": "\\frac{1}{4}",
  "\u2155": "\\frac{1}{5}",
  "\u2156": "\\frac{2}{5}",
  "\u2157": "\\frac{3}{5}",
  "\u2158": "\\frac{4}{5}",
  "\u2159": "\\frac{1}{6}",
  "\u215A": "\\frac{5}{6}",
  "\u2150": "\\frac{1}{7}",
  "\u215B": "\\frac{1}{8}",
  "\u215C": "\\frac{3}{8}",
  "\u215D": "\\frac{5}{8}",
  "\u215E": "\\frac{7}{8}",
  "\u2151": "\\frac{1}{9}",
  "\u2152": "\\frac{1}{10}",
  "\xB1": "\\pm",
  "\xD7": "\\times",
  "\u22C6": "\\star",
  \u0391: "A",
  \u03B1: "\\alpha",
  \u0392: "B",
  \u03B2: "\\beta",
  \u00DF: "\\beta",
  \u0393: "\\Gamma",
  \u03B3: "\\gamma",
  \u0394: "\\Delta",
  "\u2206": "\\Delta",
  \u03B4: "\\delta",
  \u0395: "E",
  \u03B5: "\\epsilon",
  \u0396: "Z",
  \u03B6: "\\zeta",
  \u0397: "H",
  \u03B7: "\\eta",
  \u0398: "\\Theta",
  \u03B8: "\\theta",
  \u03D1: "\\vartheta",
  \u0399: "I",
  \u03B9: "\\iota",
  \u039A: "K",
  \u03BA: "\\kappa",
  \u039B: "\\Lambda",
  \u03BB: "\\lambda",
  \u039C: "M",
  \u03BC: "\\mu",
  \u039D: "N",
  \u03BD: "\\nu",
  \u039E: "\\Xi",
  \u03BE: "\\xi",
  \u039F: "O",
  \u03BF: "o",
  \u03A0: "\\Pi",
  \u03C0: "\\pi",
  \u03A1: "P",
  \u03C1: "\\rho",
  \u03A3: "\\Sigma",
  \u03C3: "\\sigma",
  \u03A4: "T",
  \u03C4: "\\tau",
  \u03A5: "\\Upsilon",
  \u03C5: "\\upsilon",
  \u03A6: "\\Phi",
  \u03D5: "\\phi",
  \u03C6: "\\varphi",
  \u03A7: "X",
  \u03C7: "\\chi",
  \u03A8: "\\Psi",
  \u03C8: "\\psi",
  \u03A9: "\\Omega",
  \u03C9: "\\omega",
  "\u2202": "\\partial",
  "\u221E": "\\infty",
  "\u221D": "\\propto",
  "\u29DC": "\\iinfin",
  "\u29DD": "\\tieinfty",
  "\u267E": "\\acidfree",
  "\u2248": "\\approx",
  "\u2260": "\\neq",
  "\u2265": "\\geq",
  "\u2264": "\\leq",
  "\u2022": "\\cdot"
  // '‰': '\\permille',
};
function hrefToLatexText(text) {
  const replacedArray = Array.from(text !== null && text !== void 0 ? text : "").map((char) => {
    if (hrefOnlyReplacements[char])
      return { kind: "text", text: hrefOnlyReplacements[char] };
    return { kind: "text", text: char };
  });
  const replaced = replacedArray.reduce((arr, next) => {
    const prev = arr.slice(-1)[0];
    if ((prev === null || prev === void 0 ? void 0 : prev.kind) === next.kind)
      prev.text += next.text;
    else
      arr.push(next);
    return arr;
  }, []).reduce((s, next) => {
    return s + next.text;
  }, "");
  return replaced;
}
function stringToLatexText(text) {
  const escaped = (text !== null && text !== void 0 ? text : "").replace(/\\ /g, BACKSLASH_SPACE).replace(/\\/g, BACKSLASH).replace(/~/g, TILDE);
  const replacedArray = Array.from(escaped).map((char) => {
    if (textReplacements[char])
      return { kind: "text", text: textReplacements[char] };
    if (mathReplacements[char])
      return { kind: "math", text: mathReplacements[char] };
    return { kind: "text", text: char };
  });
  const replaced = replacedArray.reduce((arr, next) => {
    const prev = arr.slice(-1)[0];
    if ((prev === null || prev === void 0 ? void 0 : prev.kind) === next.kind)
      prev.text += next.text;
    else
      arr.push(next);
    return arr;
  }, []).reduce((s, next) => {
    if (next.kind === "math")
      return `${s}$${next.text}$`;
    return s + next.text;
  }, "");
  const final = replaced.replace(new RegExp(BACKSLASH_SPACE, "g"), "{\\textbackslash}~").replace(new RegExp(BACKSLASH, "g"), "{\\textbackslash}").replace(new RegExp(TILDE, "g"), "{\\textasciitilde}");
  return cleanWhitespaceChars(final, "~");
}
function stringToLatexMath(text) {
  const replaced = Array.from(text !== null && text !== void 0 ? text : "").reduce((s, char) => {
    if (mathReplacements[char]) {
      const space = s.slice(-1) === " " ? "" : " ";
      return `${s}${space}${mathReplacements[char]}`;
    }
    return s + char;
  }, "");
  const final = replaced.trim();
  return cleanWhitespaceChars(final);
}
function getLatexImageWidth(width) {
  if (typeof width === "number" && Number.isNaN(width)) {
    return getLatexImageWidth(DEFAULT_IMAGE_WIDTH);
  }
  if (typeof width === "string") {
    if (width.endsWith("%")) {
      return getLatexImageWidth(Number(width.replace("%", "")));
    } else if (width.endsWith("px")) {
      return getLatexImageWidth(Number(width.replace("px", "")) / DEFAULT_PAGE_WIDTH_PIXELS);
    }
    console.log(`Unknown width ${width} in getLatexImageWidth`);
    return getLatexImageWidth(DEFAULT_IMAGE_WIDTH);
  }
  let lineWidth = width !== null && width !== void 0 ? width : DEFAULT_IMAGE_WIDTH;
  if (lineWidth < 1)
    lineWidth *= 100;
  return `${lineWidth / 100}\\linewidth`;
}
function getClasses(className) {
  var _a;
  const classes = (_a = className === null || className === void 0 ? void 0 : className.split(" ").map((s) => s.trim().toLowerCase()).filter((s) => !!s)) !== null && _a !== void 0 ? _a : [];
  return Array.from(new Set(classes));
}
function addIndexEntries(node, state) {
  var _a;
  if (!((_a = node.indexEntries) === null || _a === void 0 ? void 0 : _a.length))
    return;
  state.data.hasIndex = true;
  node.indexEntries.forEach(({ entry, subEntry, emphasis }) => {
    let indexString = entry;
    if (subEntry === null || subEntry === void 0 ? void 0 : subEntry.value) {
      if ((subEntry === null || subEntry === void 0 ? void 0 : subEntry.kind) === "see") {
        indexString += `|see{${subEntry.value}}`;
      } else if ((subEntry === null || subEntry === void 0 ? void 0 : subEntry.kind) === "seealso") {
        indexString += `|seealso{${subEntry.value}}`;
      } else {
        indexString += `!${subEntry.value}`;
      }
    }
    if (emphasis) {
      indexString += "|textbf";
    }
    state.write(`\\index{${indexString}}`);
  });
}

// ../../node_modules/myst-to-tex/dist/tables.js
var TOTAL_TABLE_WIDTH = 886;
function renderPColumn(width) {
  if (width === 1)
    return `p{\\dimexpr \\linewidth-2\\tabcolsep}`;
  return `p{\\dimexpr ${width.toFixed(3)}\\linewidth-2\\tabcolsep}`;
}
function getColumnWidths(node) {
  var _a, _b, _c;
  let bestMaybeWidths = [];
  let mostNonNulls = 0;
  for (let i = 0; i < node.children.length; i += 1) {
    const row = node.children[i];
    const maybeWidths = row.children.reduce((acc, cell) => {
      var _a2, _b2;
      const colwidth = new Array((_a2 = cell.colspan) !== null && _a2 !== void 0 ? _a2 : 1).fill(cell.width ? cell.width / ((_b2 = cell.colspan) !== null && _b2 !== void 0 ? _b2 : 1) : null);
      return [...acc, ...colwidth];
    }, []);
    const nonNulls = maybeWidths.filter((maybeWidth) => maybeWidth > 0).length;
    if (i === 0 || nonNulls >= mostNonNulls) {
      mostNonNulls = nonNulls;
      bestMaybeWidths = maybeWidths;
      if (mostNonNulls === maybeWidths.length) {
        break;
      }
    }
  }
  let widths;
  if (mostNonNulls === bestMaybeWidths.length) {
    widths = bestMaybeWidths;
  } else {
    const totalDefinedWidths = bestMaybeWidths.reduce((acc, cur) => cur == null ? acc : acc + cur, 0);
    const remainingSpace = TOTAL_TABLE_WIDTH - totalDefinedWidths;
    const nullCells = bestMaybeWidths.length - mostNonNulls;
    const defaultWidth = Math.floor(remainingSpace / nullCells);
    widths = bestMaybeWidths.map((w) => w == null || w === 0 ? defaultWidth : w);
  }
  const total = widths.reduce((acc, cur) => acc + cur, 0);
  const fractionalWidths = widths.map((w) => w / total);
  const columnSpec = fractionalWidths.map((w) => renderPColumn(w)).join("");
  const numColumns = widths.length > 0 ? widths.length : (_c = (_b = (_a = node === null || node === void 0 ? void 0 : node.children[0]) === null || _a === void 0 ? void 0 : _a.children) === null || _b === void 0 ? void 0 : _b.length) !== null && _c !== void 0 ? _c : 0;
  return { widths: fractionalWidths, columnSpec, numColumns };
}
function renderTableCell(state, cell, i, spanIdx, widths, childCount) {
  var _a;
  let renderedSpan = 1;
  const colspan = (_a = cell.colspan) !== null && _a !== void 0 ? _a : 1;
  if (colspan > 1) {
    let width = 0;
    for (let j = 0; j < colspan; j += 1) {
      width += widths[spanIdx + j];
    }
    state.write(`\\multicolumn{${colspan}}{${renderPColumn(width)}}{`);
    renderedSpan = colspan;
  }
  if (cell.children.length === 1 && cell.children[0].type === "paragraph") {
    state.renderChildren(cell.children[0], true);
  } else {
    state.renderChildren(cell, true);
  }
  if (colspan > 1)
    state.write("}");
  if (i < childCount - 1) {
    state.write(" & ");
  }
  return renderedSpan;
}
function renderNodeToLatex(node, state) {
  state.usePackages("booktabs");
  const { widths, columnSpec, numColumns } = getColumnWidths(node);
  if (!numColumns) {
    throw new Error("invalid table format, no columns");
  }
  addIndexEntries(node, state);
  state.data.isInTable = true;
  if (!state.data.isInContainer) {
    state.write("\\bigskip\\noindent");
  }
  state.ensureNewLine();
  let numHeaderRowsFound = 0;
  if (state.data.longFigure) {
    state.ensureNewLine();
    state.write("\\hline");
    state.ensureNewLine();
    let endHeader = false;
    node.children.forEach(({ children: rowContent }) => {
      var _a, _b;
      if (endHeader)
        return;
      if ((_a = rowContent[0]) === null || _a === void 0 ? void 0 : _a.header) {
        numHeaderRowsFound += 1;
        let spanIdx = 0;
        rowContent.forEach((cell, i) => {
          spanIdx += renderTableCell(state, cell, i, spanIdx, widths, rowContent.length);
        });
        state.write(" \\\\");
        state.ensureNewLine();
      }
      if (!((_b = rowContent[0]) === null || _b === void 0 ? void 0 : _b.header)) {
        endHeader = true;
      }
    });
    if (numHeaderRowsFound > 0) {
      state.ensureNewLine();
      state.write("\\hline");
      state.ensureNewLine();
      state.write("\\endfirsthead");
      state.ensureNewLine();
      state.write("\\hline");
      state.ensureNewLine();
      state.write(`\\multicolumn{${numColumns}}{c}{\\tablename\\ \\thetable\\ -- \\textit{Continued from previous page}}\\\\`);
      state.ensureNewLine();
      node.children.forEach(({ children: rowContent }, index) => {
        if (index >= numHeaderRowsFound)
          return;
        let spanIdx = 0;
        rowContent.forEach((cell, i) => {
          spanIdx += renderTableCell(state, cell, i, spanIdx, widths, rowContent.length);
        });
        state.write(" \\\\");
        state.ensureNewLine();
      });
      state.ensureNewLine();
      state.write("\\hline");
      state.ensureNewLine();
      state.write("\\endhead");
      state.ensureNewLine();
    }
  } else {
    state.write(`\\begin{tabular}{${columnSpec}}`);
    state.ensureNewLine();
    state.write(`\\toprule`);
    state.ensureNewLine();
  }
  node.children.forEach(({ children: rowContent }, index) => {
    var _a;
    if (index < numHeaderRowsFound)
      return;
    let spanIdx = 0;
    rowContent.forEach((cell, i) => {
      spanIdx += renderTableCell(state, cell, i, spanIdx, widths, rowContent.length);
    });
    state.write(" \\\\");
    state.ensureNewLine();
    if ((_a = rowContent[0]) === null || _a === void 0 ? void 0 : _a.header) {
      state.write("\\hline");
      state.ensureNewLine();
    }
  });
  if (state.data.longFigure) {
    state.write("\\hline");
  } else {
    state.write("\\bottomrule");
    state.ensureNewLine();
    state.write("\\end{tabular}");
  }
  state.closeBlock(node);
  state.data.isInTable = false;
  if (!state.data.isInContainer) {
    state.write("\\bigskip");
  }
}

// ../../node_modules/myst-to-tex/dist/container.js
var CaptionKind;
(function(CaptionKind2) {
  CaptionKind2["fig"] = "fig";
  CaptionKind2["eq"] = "eq";
  CaptionKind2["code"] = "code";
  CaptionKind2["table"] = "table";
})(CaptionKind || (CaptionKind = {}));
function getClasses2(className) {
  var _a;
  const classes = (_a = className === null || className === void 0 ? void 0 : className.split(" ").map((s) => s.trim().toLowerCase()).filter((s) => !!s)) !== null && _a !== void 0 ? _a : [];
  return [...new Set(classes)];
}
function switchKind(node) {
  switch (node.type) {
    case "iframe":
    case "image":
      return CaptionKind.fig;
    case "table":
      return CaptionKind.table;
    case "code":
      return CaptionKind.code;
    case "math":
      return CaptionKind.eq;
    default:
      return null;
  }
}
function determineCaptionKind(node) {
  var _a;
  let kind = switchKind(node);
  (_a = node.children) === null || _a === void 0 ? void 0 : _a.forEach((n) => {
    const nKind = determineCaptionKind(n);
    if (!kind) {
      kind = nKind;
    } else if (nKind) {
      kind = CaptionKind.fig;
    }
  });
  return kind;
}
function nodeToCommand(node) {
  const kind = determineCaptionKind(node);
  const classes = getClasses2(node.class);
  const fullWidth = classes.includes("full-width") || classes.includes("w-full");
  switch (kind) {
    case CaptionKind.fig:
      return fullWidth ? "figure*" : "figure";
    case CaptionKind.table:
      return fullWidth ? "table*" : "table";
    case CaptionKind.code:
      return "figure";
    case CaptionKind.eq:
      return "figure";
    default:
      return "figure";
  }
}
function nodeToLaTeXOptions(node) {
  const kind = determineCaptionKind(node);
  switch (kind) {
    case CaptionKind.fig:
    case CaptionKind.table:
      return "!htbp";
    case CaptionKind.code:
      return "h";
    case CaptionKind.eq:
    default:
      return void 0;
  }
}
var containerHandler = (node, state) => {
  var _a;
  if (state.data.isInTable) {
    state.renderChildren(node);
    return;
  }
  const table = select("table", node);
  const containsTable = !!table;
  let tableInfo;
  if (table && node.multipage) {
    tableInfo = getColumnWidths(table);
  }
  let before;
  let after;
  if (node.landscape) {
    state.usePackages("pdflscape");
    before = "\\begin{landscape}";
    after = "\\end{landscape}";
  }
  const { enumerated, label, identifier, multipage } = node;
  const localId = (_a = label !== null && label !== void 0 ? label : identifier) !== null && _a !== void 0 ? _a : void 0;
  const command = containsTable && multipage ? "longtable" : nodeToCommand(node);
  if (command === "longtable")
    state.usePackages("longtable");
  const commandOpts = containsTable && tableInfo ? tableInfo.columnSpec : void 0;
  const bracketOpts = containsTable ? void 0 : nodeToLaTeXOptions(node);
  if (before)
    state.write(before);
  const optsInCommand = commandOpts ? `{${commandOpts}}` : "";
  const optsInBrackets = bracketOpts ? `[${bracketOpts}]` : "";
  state.write(`\\begin{${command}}${optsInCommand}${optsInBrackets}
`);
  if (!multipage || !containsTable)
    state.write("\\centering");
  state.ensureNewLine();
  state.data.longFigure = multipage;
  const lastContainer = state.data.isInContainer;
  state.data.isInContainer = true;
  state.data.nextCaptionNumbered = enumerated !== null && enumerated !== void 0 ? enumerated : !!localId;
  state.data.nextCaptionId = localId;
  state.renderChildren(node);
  state.trimEnd();
  state.data.longFigure = void 0;
  state.data.isInContainer = lastContainer;
  state.write(`
\\end{${command}}`);
  if (after)
    state.write(after);
  addIndexEntries(node, state);
  state.closeBlock(node);
};
var captionHandler = (node, state) => {
  if (state.data.isInTable && node.type !== CaptionKind.table) {
    return null;
  }
  state.ensureNewLine(true);
  const { nextCaptionNumbered: numbered, nextCaptionId: id } = state.data;
  const command = numbered === false ? "caption*" : "caption[]";
  const after = numbered && id ? `\\label{${id}}` : "";
  state.renderInlineEnvironment(node, command, { after });
};

// ../../node_modules/myst-to-tex/dist/math.js
var TOP_LEVEL_ENVIRONMENTS = [
  "equation",
  "multline",
  "gather",
  "align",
  "alignat",
  "flalign",
  "eqnarray"
];
var RE_OPEN = new RegExp(`^\\\\begin{(${TOP_LEVEL_ENVIRONMENTS.join("|")})([*]?)}`);
function isTopLevelAmsmathEnvironment(value) {
  const matchOpen = value.trim().match(RE_OPEN);
  if (!matchOpen)
    return false;
  const [, environment, star] = matchOpen;
  const end = `\\end{${environment}${star}}`;
  const matchClose = value.trim().endsWith(end);
  if (!matchClose)
    return false;
  return true;
}
function addMacrosToState(value, state) {
  if (!state.options.math)
    return;
  Object.entries(state.options.math).forEach(([k, v]) => {
    if (value.includes(k))
      state.data.mathPlugins[k] = v.macro;
  });
}
function withRecursiveCommands(state, plugins = state.data.mathPlugins) {
  if (!state.options.math)
    return plugins;
  const pluginsList = Object.entries(plugins);
  const addedPlugins = {};
  Object.entries(state.options.math).forEach(([k, v]) => {
    if (plugins[k])
      return;
    pluginsList.forEach(([, value]) => {
      if (value.includes(k))
        addedPlugins[k] = v.macro;
    });
  });
  const newPlugins = { ...addedPlugins, ...plugins };
  if (Object.keys(addedPlugins).length === 0)
    return newPlugins;
  return withRecursiveCommands(state, newPlugins);
}
var math = (node, state) => {
  const { label, enumerated } = node;
  const tightBefore = node.tight === true || node.tight === "before";
  const tightAfter = node.tight === true || node.tight === "after";
  if (tightBefore) {
    state.ensureNewLine(true);
  }
  state.usePackages("amsmath");
  addMacrosToState(node.value, state);
  addIndexEntries(node, state);
  if (state.data.isInTable) {
    state.write("\\(\\displaystyle ");
    state.write(node.value);
    state.write(" \\)");
  } else {
    const isAmsMath = isTopLevelAmsmathEnvironment(node.value);
    if (isAmsMath) {
      state.ensureNewLine();
      state.write(node.value);
      state.ensureNewLine(true);
    } else {
      state.write(`\\begin{equation${enumerated === false ? "*" : ""}}
`);
      if (label) {
        state.write(`\\label{${label}}`);
      }
      state.ensureNewLine();
      state.write(node.value);
      state.ensureNewLine(true);
      state.write(`\\end{equation${enumerated === false ? "*" : ""}}`);
    }
  }
  if (state.data.isInTable)
    return;
  if (tightAfter) {
    state.ensureNewLine(true);
  } else {
    state.closeBlock(node);
  }
};
var inlineMath = (node, state) => {
  state.usePackages("amsmath");
  addMacrosToState(node.value, state);
  state.write("$");
  state.text(node.value, true);
  state.write("$");
};
var MATH_HANDLERS = { math, inlineMath };
var math_default = MATH_HANDLERS;

// ../../node_modules/myst-to-tex/dist/legends.js
function transformLegends(mdast) {
  const containers = selectAll("container", mdast);
  containers.forEach((container) => {
    var _a;
    const captionsAndLegends = (_a = container.children) === null || _a === void 0 ? void 0 : _a.filter((child) => {
      return child.type === "caption" || child.type === "legend";
    });
    if (!(captionsAndLegends === null || captionsAndLegends === void 0 ? void 0 : captionsAndLegends.length))
      return;
    captionsAndLegends[0].type = "caption";
    captionsAndLegends.slice(1).forEach((node) => {
      var _a2;
      if (captionsAndLegends[0].children && node.children) {
        (_a2 = captionsAndLegends[0].children) === null || _a2 === void 0 ? void 0 : _a2.push(...node.children);
      }
      node.type = "__delete__";
    });
  });
  remove(mdast, "__delete__");
}

// ../../node_modules/myst-to-tex/dist/proof.js
function kindToEnvironment(kind) {
  switch (kind) {
    case "theorem":
      return "theorem";
    case "proof":
      return "proof";
    case "proposition":
      return "proposition";
    case "definition":
      return "definition";
    case "example":
      return "example";
    case "remark":
      return "remark";
    case "axiom":
      return "axiom";
    case "conjecture":
      return "conjecture";
    case "lemma":
      return "lemma";
    case "observation":
      return "observation";
    case "corollary":
      return "corollary";
    default:
      return "";
  }
}
var proofHandler = (node, state) => {
  var _a;
  state.usePackages("amsthm");
  const p = node;
  const env = kindToEnvironment((_a = p.kind) !== null && _a !== void 0 ? _a : "proof");
  if (!env) {
    fileError(state.file, `Unhandled LaTeX proof environment "${p.kind}"`, {
      node,
      source: "myst-to-tex",
      ruleId: RuleId.texRenders
    });
    return;
  }
  const t = select("admonitionTitle > text", node);
  if (t) {
    t.type = "__delete__";
  }
  const newNode = remove(node, "__delete__");
  addIndexEntries(node, state);
  state.write("\\begin{");
  state.write(env);
  state.write("}");
  if (t) {
    state.write("[");
    state.write(t.value);
    state.write("]");
  }
  if (newNode.identifier && newNode.identifier.length > 0) {
    state.write("\\label{");
    state.write(newNode.identifier);
    state.write("}");
  }
  state.renderChildren(newNode, true);
  state.write("\\end{");
  state.write(env);
  state.write("}");
  state.data.hasProofs = true;
};
var TexProofSerializer = class {
  constructor() {
    this.preamble = this.renderThmDefinitions();
  }
  renderThmDefinitions() {
    const definitions = [
      "\\newtheorem{theorem}{Theorem}[section]",
      "\\newtheorem{corollary}{Corollary}[theorem]",
      "\\newtheorem{lemma}[theorem]{Lemma}",
      "\\newtheorem{proposition}{Proposition}[section]",
      "\\newtheorem{definition}{Definition}[section]",
      "\\newtheorem{example}{Example}[section]",
      "\\newtheorem{remark}{Remark}[section]",
      "\\newtheorem{axiom}{Axiom}[section]",
      "\\newtheorem{conjecture}{Conjecture}[section]",
      "\\newtheorem{observation}{Observation}[section]"
    ];
    const block = writeTexLabelledComment("theorem", definitions, TexProofSerializer.COMMENT_LENGTH);
    const percents = "".padEnd(TexProofSerializer.COMMENT_LENGTH, "%");
    return `${percents}
${block}${percents}
`;
  }
};
TexProofSerializer.COMMENT_LENGTH = 50;

// ../../node_modules/myst-to-tex/dist/preamble.js
var TexGlossaryAndAcronymSerializer = class {
  constructor(glossaryDefinitions, acronymDefinitions) {
    const withGlossary = Object.keys(glossaryDefinitions).length > 0;
    const withAcronym = Object.keys(acronymDefinitions).length > 0;
    if (!withGlossary && !withAcronym) {
      this.printedDefinitions = "";
      this.preamble = "";
    } else {
      this.printedDefinitions = this.renderGlossary();
      this.preamble = [
        this.renderCommonImports(withAcronym),
        this.renderImports("glossary", this.createGlossaryDirectives(glossaryDefinitions)),
        this.renderImports("acronyms", this.createAcronymDirectives(acronymDefinitions))
      ].filter((item) => !!item).join("\n");
    }
  }
  renderGlossary() {
    const block = writeTexLabelledComment(
      "acronyms & glossary",
      ["\\printglossaries"],
      // Will print both glossary and abbreviations
      TexGlossaryAndAcronymSerializer.COMMENT_LENGTH
    );
    const percents = "".padEnd(TexGlossaryAndAcronymSerializer.COMMENT_LENGTH, "%");
    return `${percents}
${block}${percents}
`;
  }
  renderCommonImports(withAcronym) {
    const usepackage = withAcronym ? "\\usepackage[acronym]{glossaries}" : "\\usepackage{glossaries}";
    const makeglossaries = "\\makeglossaries";
    return `${usepackage}
${makeglossaries}
`;
  }
  renderImports(commentTitle, directives) {
    if (!directives)
      return "";
    const block = writeTexLabelledComment(commentTitle, directives, TexGlossaryAndAcronymSerializer.COMMENT_LENGTH);
    if (!block)
      return;
    const percents = "".padEnd(TexGlossaryAndAcronymSerializer.COMMENT_LENGTH, "%");
    return `${percents}
${block}${percents}
`;
  }
  createGlossaryDirectives(glossaryDefinitions) {
    const directives = Object.keys(glossaryDefinitions).map((k) => ({
      key: k,
      name: glossaryDefinitions[k][0],
      description: glossaryDefinitions[k][1]
    }));
    const entries = directives.map((entry) => `\\newglossaryentry{${entry.key}}{name=${entry.name},description={${entry.description}}}`);
    return entries;
  }
  createAcronymDirectives(acronymDefinitions) {
    const directives = Object.keys(acronymDefinitions).map((k) => ({
      key: k,
      acronym: acronymDefinitions[k][0],
      expansion: acronymDefinitions[k][1]
    }));
    return directives.map((entry) => `\\newacronym{${entry.key}}{${entry.acronym}}{${entry.expansion}}`);
  }
};
TexGlossaryAndAcronymSerializer.COMMENT_LENGTH = 50;
function generatePreamble(data) {
  const preambleLines = [];
  let suffix = "";
  if (data.hasProofs) {
    preambleLines.push(new TexProofSerializer().preamble);
  }
  if (data.hasIndex) {
    preambleLines.push("\\makeindex");
  }
  if (data.printGlossaries) {
    const glossaryState = new TexGlossaryAndAcronymSerializer(data.glossary, data.abbreviations);
    preambleLines.push(glossaryState.preamble);
    if (glossaryState.printedDefinitions) {
      suffix = `
${glossaryState.printedDefinitions}`;
    }
  }
  return { preamble: preambleLines.join("\n"), suffix };
}
function mergePreambles(current, next, warningLogFn) {
  const hasProofs = current.hasProofs || next.hasProofs;
  const hasIndex = current.hasIndex || next.hasIndex;
  const printGlossaries = current.printGlossaries || next.printGlossaries;
  Object.keys(next.glossary).forEach((key) => {
    if (Object.keys(current.glossary).includes(key)) {
      warningLogFn(`duplicate glossary entry for '${key}'`);
    }
  });
  Object.keys(next.abbreviations).forEach((key) => {
    if (Object.keys(current.abbreviations).includes(key)) {
      warningLogFn(`duplicate abbreviation definition for '${key}'`);
    }
  });
  const glossary = { ...next.glossary, ...current.glossary };
  const abbreviations = { ...next.abbreviations, ...current.abbreviations };
  return { hasProofs, hasIndex, printGlossaries, glossary, abbreviations };
}

// ../../node_modules/myst-to-tex/dist/index.js
var glossaryReferenceHandler = (node, state) => {
  if (!state.printGlossary) {
    state.renderChildren(node, true);
    return;
  }
  if (!node.identifier) {
    state.renderChildren(node, true);
    return;
  }
  const entry = state.glossary[node.identifier];
  if (!entry) {
    if (node.identifier.startsWith("index-heading-")) {
      fileWarn(state.file, `Cannot cross-reference index headings in tex export "${node.identifier}"`, {
        node,
        source: "myst-to-tex",
        ruleId: RuleId.texRenders
      });
    } else {
      fileError(state.file, `Unknown glossary entry identifier "${node.identifier}"`, {
        node,
        source: "myst-to-tex",
        ruleId: RuleId.texRenders
      });
    }
    const gn = node;
    state.write(toText(node).trim() || gn.label || "");
    return;
  }
  state.write("\\gls{");
  state.write(node.identifier);
  state.write("}");
};
var createFootnoteDefinitions = (tree) => Object.fromEntries(selectAll("footnoteDefinition", tree).map((node) => {
  const fn = node;
  return [fn.identifier, fn];
}));
var createGlossaryDefinitions = (tree) => Object.fromEntries(selectAll("glossary > definitionList > *", tree).map((node, i, siblings) => {
  if (node.type !== "definitionTerm") {
    return [];
  }
  const dt = node;
  if (!dt.identifier) {
    return [];
  }
  const dd = siblings[i + 1];
  if (dd === void 0 || dd.type !== "definitionDescription") {
    throw new Error(`Definition term has no associated description`);
  }
  const termText = toText(dt);
  const descriptionText = toText(dd);
  return [dt.identifier, [termText, descriptionText]];
}).filter((x) => x.length > 0));
var createAcronymRef = (input) => input.trim().toLowerCase();
var createAcronymDefinitions = (tree) => (
  // Note: Abbreviations contain their resolved values, hence there
  //       can be many duplicates which will be collapsed when
  //       creating the dictionary
  Object.fromEntries(selectAll("abbreviation", tree).map((node) => {
    const a = node;
    const acronymText = toText(a);
    if (!acronymText || !a.title) {
      return [];
    }
    const key = createAcronymRef(acronymText);
    return [key, [acronymText, a.title]];
  }).filter((x) => x.length > 0))
);
var handlers = {
  text(node, state) {
    state.text(node.value);
  },
  paragraph(node, state) {
    addIndexEntries(node, state);
    state.renderChildren(node);
  },
  heading(node, state) {
    const { depth, label, enumerated } = node;
    if (state.data.nextHeadingIsFrameTitle) {
      state.write("\\frametitle{");
      state.data.nextHeadingIsFrameTitle = false;
    } else {
      const star = enumerated !== false || state.options.beamer ? "" : "*";
      if (depth === -1)
        state.write(`\\part${star}{`);
      if (depth === 0)
        state.write(`\\chapter${star}{`);
      if (depth === 1)
        state.write(`\\section${star}{`);
      if (depth === 2)
        state.write(`\\subsection${star}{`);
      if (depth === 3)
        state.write(`\\subsubsection${star}{`);
      if (depth === 4)
        state.write(`\\paragraph${star}{`);
      if (depth === 5)
        state.write(`\\subparagraph${star}{`);
      if (depth === 6)
        state.write(`\\subparagraph${star}{`);
    }
    state.renderChildren(node, true);
    state.write("}");
    if (enumerated !== false && label && !node.implicit) {
      state.write(`\\label{${label}}`);
    }
    addIndexEntries(node, state);
    state.closeBlock(node);
  },
  block(node, state) {
    var _a, _b, _c;
    const metadataTags = getMetadataTags(node);
    if (state.options.beamer) {
      if (metadataTags.includes("outline")) {
        state.data.nextHeadingIsFrameTitle = false;
        state.renderChildren(node, false);
        return;
      }
      if (((_b = (_a = node.children) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.type) === "heading") {
        state.data.nextHeadingIsFrameTitle = true;
      }
      state.write("\n\n\\begin{frame}\n");
      state.renderChildren(node, false);
      state.write("\\end{frame}\n\n");
      return;
    }
    if (node.visibility === "remove" || node.visibility === "hide")
      return;
    if (metadataTags.includes("no-tex"))
      return;
    if (metadataTags.includes("no-pdf"))
      return;
    if (metadataTags.includes("new-page")) {
      state.write("\\newpage\n");
    } else if (metadataTags.includes("page-break")) {
      state.write("\\pagebreak\n");
    }
    if (((_c = node.data) === null || _c === void 0 ? void 0 : _c.part) === "index") {
      state.data.hasIndex = true;
      state.usePackages("imakeidx");
      state.write("\\printindex\n");
      return;
    }
    addIndexEntries(node, state);
    state.renderChildren(node, false);
  },
  blockquote(node, state) {
    addIndexEntries(node, state);
    state.renderEnvironment(node, "quote");
  },
  definitionList(node, state) {
    state.write("\\begin{description}\n");
    state.renderChildren(node, true);
    state.ensureNewLine();
    state.write("\\end{description}");
    state.closeBlock(node);
  },
  definitionTerm(node, state) {
    state.ensureNewLine();
    state.write("\\item[");
    state.renderChildren(node, true);
    state.write("] ");
  },
  definitionDescription(node, state) {
    state.renderChildren(node, true);
  },
  code(node, state) {
    var _a;
    if (node.visibility === "remove" || node.visibility === "hide")
      return;
    addIndexEntries(node, state);
    let start = "\\begin{verbatim}\n";
    let end = "\n\\end{verbatim}";
    if (state.options.codeStyle === "listings" || getClasses(node.class).includes("listings") && node.lang !== void 0) {
      state.usePackages("listings");
      start = `\\begin{lstlisting}[language=${node.lang}]
`;
      end = "\n\\end{lstlisting}";
    } else if (state.options.codeStyle === "minted" || getClasses(node.class).includes("minted")) {
      state.usePackages("minted");
      start = `\\begin{minted}[breaklines]{${(_a = node.lang) !== null && _a !== void 0 ? _a : "text"}}
`;
      end = "\n\\end{minted}";
    }
    state.write(start);
    state.text(node.value, true);
    state.write(end);
    state.closeBlock(node);
  },
  list(node, state) {
    addIndexEntries(node, state);
    if (state.data.isInTable) {
      node.children.forEach((child, i) => {
        state.write(node.ordered ? `${i}.~~` : "\\textbullet~~");
        state.renderChildren(child, true);
        state.write("\\newline");
        state.ensureNewLine();
      });
    } else {
      state.renderEnvironment(node, node.ordered ? "enumerate" : "itemize", {
        parameters: node.ordered && node.start !== 1 ? "resume" : void 0
      });
    }
  },
  listItem(node, state) {
    var _a, _b;
    state.write("\\item ");
    if (((_b = (_a = node.children) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.type) === "paragraph" && node.children.length === 1) {
      state.renderChildren(node.children[0], true);
    } else {
      state.renderChildren(node, true);
    }
    state.write("\n");
  },
  thematicBreak(node, state) {
    state.write("\n\\bigskip\n\\centerline{\\rule{13cm}{0.4pt}}\n\\bigskip");
    state.closeBlock(node);
  },
  ...math_default,
  mystRole(node, state) {
    state.renderChildren(node, true);
  },
  mystDirective(node, state) {
    state.renderChildren(node, false);
  },
  div(node, state) {
    addIndexEntries(node, state);
    state.renderChildren(node, false);
  },
  span(node, state) {
    state.renderChildren(node, true);
    addIndexEntries(node, state);
  },
  comment(node, state) {
    var _a, _b;
    state.ensureNewLine();
    state.write(`% ${(_b = (_a = node.value) === null || _a === void 0 ? void 0 : _a.split("\n").join("\n% ")) !== null && _b !== void 0 ? _b : ""}`);
    state.closeBlock(node);
  },
  strong(node, state) {
    state.renderInlineEnvironment(node, "textbf");
  },
  emphasis(node, state) {
    state.renderInlineEnvironment(node, "textit");
  },
  underline(node, state) {
    state.renderInlineEnvironment(node, "uline");
  },
  inlineCode(node, state) {
    state.write("\\texttt{");
    state.text(node.value, false);
    state.write("}");
  },
  subscript(node, state) {
    state.renderInlineEnvironment(node, "textsubscript");
  },
  superscript(node, state) {
    state.renderInlineEnvironment(node, "textsuperscript");
  },
  delete(node, state) {
    state.usePackages("ulem");
    state.renderInlineEnvironment(node, "sout");
  },
  break(node, state) {
    state.write("\\newline");
    state.ensureNewLine();
  },
  abbreviation(node, state) {
    if (!state.printGlossary) {
      state.renderChildren(node, true);
      return;
    }
    const acronymText = toText(node);
    if (!acronymText) {
      return [];
    }
    const ref = createAcronymRef(acronymText);
    const entry = state.abbreviations[ref];
    if (!entry) {
      fileError(state.file, `Unknown abbreviation entry identifier "${ref}"`, {
        node,
        source: "myst-to-tex",
        ruleId: RuleId.texRenders
      });
      return;
    }
    state.write("\\acrshort{");
    state.write(ref);
    state.write("}");
  },
  glossary() {
  },
  link(node, state) {
    var _a;
    state.usePackages("url", "hyperref");
    const href = node.url;
    if (((_a = node.children[0]) === null || _a === void 0 ? void 0 : _a.value) === href) {
      state.write("\\url{");
      state.write(hrefToLatexText(href));
      state.write("}");
      return;
    }
    state.write("\\href{");
    state.write(hrefToLatexText(href));
    state.write("}{");
    state.renderChildren(node, true);
    state.write("}");
  },
  admonition(node, state) {
    addIndexEntries(node, state);
    state.usePackages("framed");
    state.renderEnvironment(node, "framed");
  },
  admonitionTitle(node, state) {
    state.renderInlineEnvironment(node, "textbf");
    state.write("\\\\\n");
  },
  table: renderNodeToLatex,
  image(node, state) {
    addIndexEntries(node, state);
    state.usePackages("graphicx");
    const { width: nodeWidth, url: nodeSrc, align: nodeAlign } = node;
    const src = nodeSrc;
    const width = getLatexImageWidth(nodeWidth);
    state.write(`\\includegraphics[width=${width}]{${src}}`);
    state.closeBlock(node);
  },
  container: containerHandler,
  proof: proofHandler,
  caption: captionHandler,
  captionNumber: () => void 0,
  crossReference(node, state, parent) {
    var _a, _b;
    if (node.kind === "definitionTerm") {
      glossaryReferenceHandler(node, state, parent);
      return;
    }
    const usedTemplate = ((_a = node.template) === null || _a === void 0 ? void 0 : _a.includes("%s")) ? node.template : void 0;
    const text = ((_b = usedTemplate !== null && usedTemplate !== void 0 ? usedTemplate : toText(node)) === null || _b === void 0 ? void 0 : _b.replace(/\s/g, "~")) || "%s";
    const id = node.label;
    state.write(text.replace(/%s/g, `\\ref{${id}}`));
  },
  citeGroup(node, state) {
    if (state.options.citestyle === "numerical-only") {
      state.write("\\cite{");
    } else if (state.options.bibliography === "biblatex") {
      const command = node.kind === "narrative" ? "textcite" : "parencite";
      state.write(`\\${command}{`);
    } else {
      const tp = node.kind === "narrative" ? "t" : "p";
      state.write(`\\cite${tp}{`);
    }
    state.renderChildren(node, true, ", ");
    state.write("}");
  },
  cite(node, state, parent) {
    if (!state.options.bibliography) {
      state.usePackages("natbib");
    }
    if (parent.type === "citeGroup") {
      state.write(node.label);
    } else if (state.options.bibliography === "biblatex") {
      state.write(`\\textcite{${node.label}}`);
    } else {
      state.write(`\\cite{${node.label}}`);
    }
  },
  embed(node, state) {
    state.renderChildren(node, true);
  },
  include(node, state) {
    state.renderChildren(node, true);
  },
  footnoteReference(node, state) {
    if (!node.identifier)
      return;
    const footnote = state.footnotes[node.identifier];
    if (!footnote) {
      fileError(state.file, `Unknown footnote identifier "${node.identifier}"`, {
        node,
        source: "myst-to-tex",
        ruleId: RuleId.texRenders
      });
      return;
    }
    state.write("\\footnote{");
    state.renderChildren(footnote, true);
    state.trimEnd();
    state.write("}");
  },
  footnoteDefinition() {
  },
  si(node, state) {
    var _a, _b, _c, _d;
    state.usePackages("siunitx");
    if (node.number == null) {
      state.write(`\\unit{${(_b = (_a = node.units) === null || _a === void 0 ? void 0 : _a.map((u) => `\\${u}`).join("")) !== null && _b !== void 0 ? _b : ""}}`);
    } else {
      state.write(`\\qty{${node.number}}{${(_d = (_c = node.units) === null || _c === void 0 ? void 0 : _c.map((u) => `\\${u}`).join("")) !== null && _d !== void 0 ? _d : ""}}`);
    }
  },
  inlineExpression(node, state) {
    var _a;
    if ((_a = node.children) === null || _a === void 0 ? void 0 : _a.length) {
      state.renderChildren(node, true);
    } else {
      state.write("\\texttt{");
      state.text(node.value, false);
      state.write("}");
    }
  },
  raw(node, state) {
    var _a;
    if (node.tex) {
      state.write(node.tex);
    } else if ((_a = node.children) === null || _a === void 0 ? void 0 : _a.length) {
      state.renderChildren(node);
    }
  },
  toc(node, state) {
    var _a;
    const title = (_a = node.children) === null || _a === void 0 ? void 0 : _a[0];
    if (title) {
      state.write("\\renewcommand{\\contentsname}{");
      state.text(toText(title));
      state.write("}\n");
    }
    if (node.depth) {
      state.write(`\\setcounter{tocdepth}{${node.depth}}
`);
    }
    state.write("\\tableofcontents\n");
  }
};
var TexSerializer = class {
  constructor(file, tree, opts) {
    var _a, _b;
    file.result = "";
    this.file = file;
    this.options = opts !== null && opts !== void 0 ? opts : {};
    this.data = { mathPlugins: {}, imports: /* @__PURE__ */ new Set() };
    this.handlers = (_a = opts === null || opts === void 0 ? void 0 : opts.handlers) !== null && _a !== void 0 ? _a : handlers;
    this.references = (_b = opts === null || opts === void 0 ? void 0 : opts.references) !== null && _b !== void 0 ? _b : {};
    this.footnotes = createFootnoteDefinitions(tree);
    this.glossary = (opts === null || opts === void 0 ? void 0 : opts.printGlossaries) ? createGlossaryDefinitions(tree) : {};
    this.abbreviations = (opts === null || opts === void 0 ? void 0 : opts.printGlossaries) ? createAcronymDefinitions(tree) : {};
    this.renderChildren(tree);
  }
  get printGlossary() {
    return this.options.printGlossaries === true;
  }
  get out() {
    return this.file.result;
  }
  usePackages(...packageNames) {
    packageNames.forEach((p) => {
      this.data.imports.add(p);
    });
  }
  write(value) {
    this.file.result += value;
  }
  text(value, mathMode = false) {
    const escaped = mathMode ? stringToLatexMath(value) : stringToLatexText(value);
    this.write(escaped);
  }
  trimEnd() {
    this.file.result = this.out.trimEnd();
  }
  ensureNewLine(trim = false) {
    if (trim)
      this.trimEnd();
    if (this.out.endsWith("\n"))
      return;
    this.write("\n");
  }
  renderChildren(node, inline = false, delim = "") {
    var _a, _b, _c;
    const numChildren = (_b = (_a = node.children) === null || _a === void 0 ? void 0 : _a.length) !== null && _b !== void 0 ? _b : 0;
    (_c = node.children) === null || _c === void 0 ? void 0 : _c.forEach((child, index) => {
      const handler = this.handlers[child.type];
      if (handler) {
        handler(child, this, node);
      } else {
        fileError(this.file, `Unhandled LaTeX conversion for node of "${child.type}"`, {
          node: child,
          source: "myst-to-tex",
          ruleId: RuleId.texRenders
        });
      }
      if (delim && index + 1 < numChildren)
        this.write(delim);
    });
    if (!inline)
      this.closeBlock(node);
  }
  renderEnvironment(node, env, opts) {
    const optsInBrackets = (opts === null || opts === void 0 ? void 0 : opts.parameters) ? `[${opts.parameters}]` : "";
    const optsInBraces = (opts === null || opts === void 0 ? void 0 : opts.arguments) ? `{${opts.arguments.join("}{")}}` : "";
    this.file.result += `\\begin{${env}}${optsInBrackets}${optsInBraces}
`;
    this.renderChildren(node, true);
    this.ensureNewLine(true);
    this.file.result += `\\end{${env}}`;
    this.closeBlock(node);
  }
  renderInlineEnvironment(node, env, opts) {
    this.file.result += `\\${env}{`;
    this.renderChildren(node, true);
    this.trimEnd();
    this.file.result += "}";
    if (opts === null || opts === void 0 ? void 0 : opts.after) {
      this.ensureNewLine(true);
      this.write(opts.after);
    }
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  closeBlock(node) {
    this.ensureNewLine(true);
    this.file.result += "\n";
  }
};
var plugin = function(opts) {
  this.Compiler = (node, file) => {
    transformLegends(node);
    const state = new TexSerializer(file, node, opts !== null && opts !== void 0 ? opts : { handlers });
    const tex = file.result.trim();
    const result = {
      imports: [...state.data.imports],
      preamble: {
        hasProofs: state.data.hasProofs,
        hasIndex: state.data.hasIndex,
        printGlossaries: opts === null || opts === void 0 ? void 0 : opts.printGlossaries,
        glossary: state.glossary,
        abbreviations: state.abbreviations
      },
      commands: withRecursiveCommands(state),
      value: tex
    };
    file.result = result;
    return file;
  };
  return (node) => {
    return node;
  };
};
var dist_default = plugin;
export {
  DEFAULT_IMAGE_WIDTH,
  DEFAULT_PAGE_WIDTH_PIXELS,
  dist_default as default,
  generatePreamble,
  mergePreambles
};
//# sourceMappingURL=/build/_shared/dist-K6SQELBU.js.map
