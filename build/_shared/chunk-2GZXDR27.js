import {
  jsYaml
} from "/build/_shared/chunk-DOEQHMLC.js";
import {
  require_classnames
} from "/build/_shared/chunk-ZNG4DSCS.js";
import {
  NotebookCell,
  RuleId,
  createIndexEntries,
  fileError,
  fileWarn,
  normalizeLabel,
  parseIndexLine,
  select,
  toText
} from "/build/_shared/chunk-Q6DHUCUI.js";
import {
  __toESM
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-directives/dist/utils.js
function classDirectiveOption(nodeType = "node") {
  return {
    class: {
      type: String,
      doc: `Annotate the ${nodeType} with a set of space-delimited class names.`
    }
  };
}
function labelDirectiveOption(nodeType = "node") {
  return {
    label: {
      type: String,
      alias: ["name"],
      doc: `Label the ${nodeType} to be cross-referenced or explicitly linked to.`
    }
  };
}
function enumerationDirectiveOptions(nodeType = "node") {
  return {
    enumerated: {
      type: Boolean,
      alias: ["numbered"],
      doc: `Turn on/off the numbering for the specific ${nodeType}`
    },
    enumerator: {
      type: String,
      alias: ["number"],
      doc: `Explicitly set the ${nodeType} number`
    }
  };
}
function commonDirectiveOptions(nodeType = "node") {
  return {
    ...classDirectiveOption(nodeType),
    ...labelDirectiveOption(nodeType),
    ...enumerationDirectiveOptions(nodeType)
  };
}
function addClassOptions(data, node) {
  var _a;
  if (typeof ((_a = data.options) === null || _a === void 0 ? void 0 : _a.class) === "string") {
    node.class = data.options.class;
  }
  return node;
}
function addLabelOptions(data, node) {
  var _a;
  const { label, identifier } = normalizeLabel((_a = data.options) === null || _a === void 0 ? void 0 : _a.label) || {};
  if (label)
    node.label = label;
  if (identifier)
    node.identifier = identifier;
  return node;
}
function addEnumerationOptions(data, node) {
  var _a, _b;
  if (typeof ((_a = data.options) === null || _a === void 0 ? void 0 : _a.enumerated) === "boolean") {
    node.enumerated = data.options.enumerated;
  }
  if ((_b = data.options) === null || _b === void 0 ? void 0 : _b.enumerator) {
    node.enumerator = data.options.enumerator;
  }
  return node;
}
function addCommonDirectiveOptions(data, node) {
  addClassOptions(data, node);
  addLabelOptions(data, node);
  addEnumerationOptions(data, node);
  return node;
}

// ../../node_modules/myst-directives/dist/admonition.js
var admonitionDirective = {
  name: "admonition",
  doc: 'Callouts, or "admonitions", highlight a particular block of text that exists slightly apart from the narrative of your page, such as a note or a warning. \n\n The admonition kind can be determined by the directive name used (e.g. `:::{tip}` or `:::{note}`).',
  alias: [
    "attention",
    "caution",
    "danger",
    "error",
    "important",
    "hint",
    "note",
    "seealso",
    "tip",
    "warning"
  ],
  arg: {
    type: "myst",
    doc: "The optional title of the admonition, if not supplied the admonition kind will be used.\n\nNote that the argument parsing is different from Sphinx, which does not allow named admonitions to have custom titles."
  },
  options: {
    ...commonDirectiveOptions("admonition"),
    class: {
      type: String,
      doc: `CSS classes to add to your admonition. Special classes include:

- \`dropdown\`: turns the admonition into a \`<details>\` html element
- \`simple\`: an admonition with "simple" styles
- the name of an admonition, the first valid admonition name encountered will be used (e.g. \`tip\`). Note that if you provide conflicting class names, the first valid admonition name will be used.`
    },
    icon: {
      type: Boolean,
      doc: "Setting icon to false will hide the icon."
      // class_option: list of strings?
    },
    open: {
      type: Boolean,
      doc: "Turn the admonition into a dropdown, if it isn't already, and set the open state."
    }
  },
  body: {
    type: "myst",
    doc: "The body of the admonition. If there is no title and the body starts with bold text or a heading, that content will be used as the admonition title."
  },
  run(data) {
    var _a, _b, _c, _d, _e;
    const children = [];
    if (data.arg) {
      children.push({
        type: data.body ? "admonitionTitle" : "paragraph",
        children: data.arg
      });
    }
    if (data.body) {
      children.push(...data.body);
    }
    const admonition = {
      type: "admonition",
      kind: data.name !== "admonition" ? data.name : void 0,
      children
    };
    if (((_a = data.options) === null || _a === void 0 ? void 0 : _a.icon) === false) {
      admonition.icon = false;
    }
    addCommonDirectiveOptions(data, admonition);
    if (typeof ((_b = data.options) === null || _b === void 0 ? void 0 : _b.open) === "boolean") {
      if (!((_c = admonition.class) === null || _c === void 0 ? void 0 : _c.split(" ").includes("dropdown"))) {
        admonition.class = `${(_d = admonition.class) !== null && _d !== void 0 ? _d : ""} dropdown`.trim();
      }
      if ((_e = data.options) === null || _e === void 0 ? void 0 : _e.open)
        admonition.open = true;
    }
    return [admonition];
  }
};

// ../../node_modules/myst-directives/dist/bibliography.js
var bibliographyDirective = {
  name: "bibliography",
  options: {
    ...commonDirectiveOptions("bibliography"),
    filter: {
      type: String
    }
  },
  run(data) {
    var _a;
    const bibliography = {
      type: "bibliography",
      filter: (_a = data.options) === null || _a === void 0 ? void 0 : _a.filter
    };
    addCommonDirectiveOptions(data, bibliography);
    return [bibliography];
  }
};

// ../../node_modules/myst-directives/node_modules/nanoid/url-alphabet/index.js
var urlAlphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict";

// ../../node_modules/myst-directives/node_modules/nanoid/index.browser.js
var nanoid = (size = 21) => {
  let id = "";
  let bytes = crypto.getRandomValues(new Uint8Array(size |= 0));
  while (size--) {
    id += urlAlphabet[bytes[size] & 63];
  }
  return id;
};

// ../../node_modules/myst-directives/dist/code.js
function parseEmphasizeLines(data, vfile, emphasizeLinesString) {
  const { node } = data;
  if (!emphasizeLinesString)
    return void 0;
  const result = /* @__PURE__ */ new Set();
  for (const part of emphasizeLinesString.split(",")) {
    const trimmed = part.trim();
    const rangeMatch = trimmed.match(/^(\d+)\s*-\s*(\d+)$/);
    if (rangeMatch) {
      const start = Number(rangeMatch[1]);
      const end = Number(rangeMatch[2]);
      if (start <= end) {
        for (let i = start; i <= end; i++) {
          result.add(i);
        }
        continue;
      } else {
        fileWarn(vfile, `Invalid emphasize-lines range "${trimmed}" (start > end)`, {
          node,
          source: "code-block:options",
          ruleId: RuleId.directiveOptionsCorrect
        });
        continue;
      }
    }
    if (/^\d+$/.test(trimmed)) {
      result.add(Number(trimmed));
    } else if (trimmed !== "") {
      fileWarn(vfile, `Invalid emphasize-lines value "${trimmed}"`, {
        node,
        source: "code-block:options",
        ruleId: RuleId.directiveOptionsCorrect
      });
    }
  }
  if (result.size === 0)
    return void 0;
  return Array.from(result).sort((a, b) => a - b);
}
function getCodeBlockOptions(data, vfile, defaultFilename) {
  var _a;
  const { options, node } = data;
  if ((options === null || options === void 0 ? void 0 : options["lineno-start"]) != null && (options === null || options === void 0 ? void 0 : options["number-lines"]) != null) {
    fileWarn(vfile, 'Cannot use both "lineno-start" and "number-lines"', {
      node: (_a = select('mystOption[name="number-lines"]', node)) !== null && _a !== void 0 ? _a : node,
      source: "code-block:options",
      ruleId: RuleId.directiveOptionsCorrect
    });
  }
  const emphasizeLines = parseEmphasizeLines(data, vfile, options === null || options === void 0 ? void 0 : options["emphasize-lines"]);
  const numberLines = options === null || options === void 0 ? void 0 : options["number-lines"];
  const showLineNumbers = (options === null || options === void 0 ? void 0 : options.linenos) || (options === null || options === void 0 ? void 0 : options["lineno-start"]) || (options === null || options === void 0 ? void 0 : options["lineno-match"]) || numberLines ? true : void 0;
  let startingLineNumber = numberLines != null && numberLines > 1 ? numberLines : options === null || options === void 0 ? void 0 : options["lineno-start"];
  if (options === null || options === void 0 ? void 0 : options["lineno-match"]) {
    startingLineNumber = "match";
  } else if (startingLineNumber == null || startingLineNumber <= 1) {
    startingLineNumber = void 0;
  }
  let filename = options === null || options === void 0 ? void 0 : options["filename"];
  if ((filename === null || filename === void 0 ? void 0 : filename.toLowerCase()) === "false") {
    filename = void 0;
  } else if (!filename && defaultFilename) {
    filename = defaultFilename;
  }
  return {
    emphasizeLines,
    showLineNumbers,
    startingLineNumber,
    filename
  };
}
var CODE_DIRECTIVE_OPTIONS = {
  caption: {
    type: "myst",
    doc: "A parsed caption for the code block."
  },
  linenos: {
    type: Boolean,
    doc: "Show line numbers"
  },
  "lineno-start": {
    type: Number,
    doc: "Start line numbering from a particular value, default is 1. If present, line numbering is activated."
  },
  "number-lines": {
    type: Number,
    doc: 'Alternative for "lineno-start", turns on line numbering and can be an integer that is the start of the line numbering.'
  },
  "emphasize-lines": {
    type: String,
    doc: 'Emphasize particular lines (comma-separated numbers which can include ranges), e.g. "3,5,7-9"'
  },
  filename: {
    type: String,
    doc: "Show the filename in addition to the rendered code. The `include` directive will use the filename by default, to turn off this default set the filename to `false`."
  }
  // dedent: {
  //   type: Number,
  //   doc: 'Strip indentation characters from the code block',
  // },
  // force: {
  //   type: Boolean,
  //   doc: 'Ignore minor errors on highlighting',
  // },
};
function parseTags(input, vfile, node) {
  var _a, _b;
  if (!input)
    return void 0;
  if (typeof input === "string" && input.startsWith("[") && input.endsWith("]")) {
    try {
      return parseTags(jsYaml.load(input), vfile, node);
    } catch (error) {
      fileError(vfile, "Could not load tags for code-cell directive", {
        node: (_a = select('mystOption[name="tags"]', node)) !== null && _a !== void 0 ? _a : node,
        source: "code-cell:tags",
        ruleId: RuleId.directiveOptionsCorrect
      });
      return void 0;
    }
  }
  if (typeof input === "string") {
    const tags2 = input.split(/[,\s]/).map((t) => t.trim()).filter((t) => !!t);
    return tags2.length > 0 ? tags2 : void 0;
  }
  if (!Array.isArray(input))
    return void 0;
  const tags = input;
  if (tags && Array.isArray(tags) && tags.every((t) => typeof t === "string")) {
    if (tags.length > 0) {
      return tags.map((t) => t.trim()).filter((t) => !!t);
    }
  } else if (tags) {
    fileWarn(vfile, "tags in code-cell directive must be a list of strings", {
      node: (_b = select('mystOption[name="tags"]', node)) !== null && _b !== void 0 ? _b : node,
      source: "code-cell:tags",
      ruleId: RuleId.directiveOptionsCorrect
    });
    return void 0;
  }
}
var codeDirective = {
  name: "code",
  doc: "A code-block environment with a language as the argument, and options for highlighting, showing line numbers, and an optional filename.",
  alias: ["code-block", "sourcecode"],
  arg: {
    type: String,
    doc: "Code language, for example `python` or `typescript`"
  },
  options: {
    ...commonDirectiveOptions("code"),
    ...CODE_DIRECTIVE_OPTIONS
  },
  body: {
    type: String,
    doc: "The raw code to display for the code block."
  },
  run(data, vfile) {
    var _a;
    const opts = getCodeBlockOptions(data, vfile);
    const code = {
      type: "code",
      lang: data.arg,
      ...opts,
      value: data.body
    };
    if (!((_a = data.options) === null || _a === void 0 ? void 0 : _a.caption)) {
      addCommonDirectiveOptions(data, code);
      return [code];
    }
    const caption = {
      type: "caption",
      children: [
        {
          type: "paragraph",
          children: data.options.caption
        }
      ]
    };
    const container = {
      type: "container",
      kind: "code",
      children: [code, caption]
    };
    addCommonDirectiveOptions(data, container);
    return [container];
  }
};
var codeCellDirective = {
  name: "code-cell",
  doc: "An executable code cell",
  arg: {
    type: String,
    doc: "Language for execution and display, for example `python`. It will default to the language of the notebook or containing markdown file."
  },
  options: {
    ...commonDirectiveOptions("code-cell"),
    caption: {
      type: "myst",
      doc: "A parsed caption for the code output."
    },
    tags: {
      type: String,
      alias: ["tag"],
      doc: "A comma-separated list of tags to add to the cell, for example, `remove-input` or `hide-cell`."
    }
  },
  body: {
    type: String,
    doc: "The code to be executed and displayed."
  },
  run(data, vfile) {
    var _a, _b, _c;
    const code = {
      type: "code",
      lang: data.arg,
      executable: true,
      value: (_a = data.body) !== null && _a !== void 0 ? _a : ""
    };
    const output = {
      type: "output",
      id: nanoid(),
      data: []
    };
    const block = {
      type: "block",
      kind: NotebookCell.code,
      children: [code, output],
      data: {}
    };
    addCommonDirectiveOptions(data, block);
    if ((_b = data.options) === null || _b === void 0 ? void 0 : _b.caption) {
      block.data.caption = [{ type: "paragraph", children: data.options.caption }];
    }
    const tags = parseTags((_c = data.options) === null || _c === void 0 ? void 0 : _c.tags, vfile, data.node);
    if (tags)
      block.data.tags = tags;
    return [block];
  }
};

// ../../node_modules/myst-directives/dist/dropdown.js
var dropdownDirective = {
  name: "dropdown",
  arg: {
    type: "myst"
  },
  options: {
    ...commonDirectiveOptions("dropdown"),
    // TODO: Add enumeration in future
    open: {
      type: Boolean,
      doc: "When true, the dropdown starts open."
    }
    // Legacy options we may want to implement:
    // color
    // icon
    // animate
    // margin
    // name
    // 'class-container'
    // 'class-title'
    // 'class-body'
  },
  body: {
    type: "myst",
    required: true
  },
  run(data) {
    var _a;
    const children = [];
    if (data.arg) {
      children.push({ type: "summary", children: data.arg });
    }
    children.push(...data.body);
    const details = {
      type: "details",
      open: (_a = data.options) === null || _a === void 0 ? void 0 : _a.open,
      children
    };
    addCommonDirectiveOptions(data, details);
    return [details];
  }
};

// ../../node_modules/myst-directives/dist/embed.js
var embedDirective = {
  name: "embed",
  doc: "The embed directive allows you to duplicate content from another part of your project. This can also be done through the figure directive.",
  arg: {
    type: String,
    doc: "The label of the node that you are embedding.",
    required: true
  },
  options: {
    ...commonDirectiveOptions("embed"),
    "remove-input": {
      type: Boolean,
      doc: "If embedding a Jupyter Notebook cell, remove the input of the cell."
    },
    "remove-output": {
      type: Boolean,
      doc: "If embedding a Jupyter Notebook cell, remove the output of the cell."
    }
  },
  run(data) {
    var _a, _b;
    if (!data.arg)
      return [];
    const argString = data.arg;
    const arg = argString.startsWith("#") ? argString.substring(1) : argString;
    const { label } = normalizeLabel(arg) || {};
    if (!label)
      return [];
    const embed = {
      type: "embed",
      source: { label },
      "remove-input": (_a = data.options) === null || _a === void 0 ? void 0 : _a["remove-input"],
      "remove-output": (_b = data.options) === null || _b === void 0 ? void 0 : _b["remove-output"]
    };
    addCommonDirectiveOptions(data, embed);
    return [embed];
  }
};

// ../../node_modules/myst-directives/dist/figure.js
var figureDirective = {
  name: "figure",
  arg: {
    type: String,
    doc: "The filename of an image (e.g. `my-fig.png`), or an ID of a Jupyter Notebook cell (e.g. `#my-cell`)."
  },
  options: {
    ...commonDirectiveOptions("figure"),
    class: {
      type: String,
      alias: ["figclass"],
      doc: `CSS classes to add to your figure. Special classes include:

- \`full-width\`: changes the figure environment to cover two columns in LaTeX`
    },
    height: {
      type: String,
      doc: "The figure height, in CSS units, for example `4em` or `300px`.",
      alias: ["h"]
    },
    width: {
      type: String,
      // TODO: validate that this is a CSS width
      alias: ["w", "figwidth"],
      doc: "The figure width, in CSS units, for example `50%` or `300px`."
    },
    alt: {
      type: String,
      doc: "Alternative text for the image"
    },
    // scale: {
    //   type: Number,
    // },
    // target: {
    //   type: String,
    // },
    align: {
      type: String,
      doc: "The alignment of the image in the figure. Choose one of `left`, `center` or `right`"
      // TODO: this is not implemented below
      // choice(["left", "center", "right"])
    },
    "remove-input": {
      type: Boolean,
      doc: "If the argument is a notebook cell, use this flag to remove the input code from the cell."
    },
    "remove-output": {
      type: Boolean,
      doc: "If the argument is a notebook cell, use this flag to remove the output from the cell."
    },
    placeholder: {
      type: String,
      doc: "A placeholder image when using a notebook cell as the figure contents. This will be shown in place of the Jupyter output until an execution environment is attached. It will also be used in static outputs, such as a PDF output."
    },
    "no-subfigures": {
      type: Boolean,
      doc: "Disallow implicit subfigure creation from child nodes",
      alias: ["no-subfig", "no-subfigure"]
    },
    kind: {
      type: String,
      doc: 'Override the figures "kind", which changes the enumeration to start counting independently for that kind. For example, `kind: "example"`. The default enumeration and referencing will be the capitalized `kind` followed by a number (e.g. "Example 1").'
    }
  },
  body: {
    type: "myst",
    doc: "If an argument is provided to the figure directive, the body will be the figure caption. You may also omit the figure directive argument and provide images in the body of the figure, these will be parsed into sub figures."
  },
  run(data) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o;
    const children = [];
    if (data.arg) {
      children.push({
        type: "image",
        url: data.arg,
        alt: (_a = data.options) === null || _a === void 0 ? void 0 : _a.alt,
        width: (_b = data.options) === null || _b === void 0 ? void 0 : _b.width,
        height: (_c = data.options) === null || _c === void 0 ? void 0 : _c.height,
        align: (_d = data.options) === null || _d === void 0 ? void 0 : _d.align,
        // These will pass through if the node is converted to an embed node in the image transform
        "remove-input": (_e = data.options) === null || _e === void 0 ? void 0 : _e["remove-input"],
        "remove-output": (_f = data.options) === null || _f === void 0 ? void 0 : _f["remove-output"]
      });
    }
    if ((_g = data.options) === null || _g === void 0 ? void 0 : _g.placeholder) {
      children.push({
        type: "image",
        placeholder: true,
        url: data.options.placeholder,
        alt: (_h = data.options) === null || _h === void 0 ? void 0 : _h.alt,
        width: (_j = data.options) === null || _j === void 0 ? void 0 : _j.width,
        height: (_k = data.options) === null || _k === void 0 ? void 0 : _k.height,
        align: (_l = data.options) === null || _l === void 0 ? void 0 : _l.align
      });
    }
    if (data.body) {
      children.push(...data.body);
    }
    const container = {
      type: "container",
      kind: ((_m = data.options) === null || _m === void 0 ? void 0 : _m.kind) || "figure",
      children
    };
    addCommonDirectiveOptions(data, container);
    if ((_o = data.options) === null || _o === void 0 ? void 0 : _o["no-subfigures"]) {
      container.noSubcontainers = true;
    }
    return [container];
  }
};

// ../../node_modules/myst-directives/dist/iframe.js
var iframeDirective = {
  name: "iframe",
  arg: {
    type: String,
    doc: "The URL source (`src`) of the HTML iframe element.",
    required: true
  },
  options: {
    ...commonDirectiveOptions("iframe"),
    width: {
      type: String,
      doc: "The iframe width, in CSS units, for example `50%` or `300px`."
    },
    align: {
      type: String,
      doc: "The alignment of the iframe in the page. Choose one of `left`, `center` or `right`"
    },
    title: {
      type: String,
      doc: "The title attribute for the iframe element, describing its content for accessibility."
    },
    placeholder: {
      type: String,
      doc: "A placeholder image for the iframe in static exports."
    }
  },
  body: { type: "myst", doc: "If provided, this will be the iframe caption." },
  run(data) {
    var _a, _b, _c, _d, _e, _f, _g, _h;
    const iframe = {
      type: "iframe",
      src: data.arg,
      width: (_a = data.options) === null || _a === void 0 ? void 0 : _a.width,
      align: (_b = data.options) === null || _b === void 0 ? void 0 : _b.align,
      title: (_c = data.options) === null || _c === void 0 ? void 0 : _c.title
    };
    if ((_d = data.options) === null || _d === void 0 ? void 0 : _d.placeholder) {
      iframe.children = [
        {
          type: "image",
          placeholder: true,
          url: data.options.placeholder,
          alt: (_e = data.options) === null || _e === void 0 ? void 0 : _e.alt,
          width: (_f = data.options) === null || _f === void 0 ? void 0 : _f.width,
          height: (_g = data.options) === null || _g === void 0 ? void 0 : _g.height,
          align: (_h = data.options) === null || _h === void 0 ? void 0 : _h.align
        }
      ];
    }
    if (!data.body) {
      addCommonDirectiveOptions(data, iframe);
      return [iframe];
    }
    const container = {
      type: "container",
      kind: "figure",
      children: [iframe, { type: "caption", children: data.body }]
    };
    addCommonDirectiveOptions(data, container);
    return [container];
  }
};

// ../../node_modules/myst-directives/dist/image.js
var imageDirective = {
  name: "image",
  arg: {
    type: String,
    doc: "The filename of an image (e.g. `my-fig.png`).",
    required: true
  },
  options: {
    ...commonDirectiveOptions("image"),
    height: {
      type: String,
      doc: "The image height, in CSS units, for example `4em` or `300px`.",
      alias: ["h"]
    },
    width: {
      type: String,
      alias: ["w"],
      doc: "The image width, in CSS units, for example `50%` or `300px`."
    },
    alt: {
      type: String,
      doc: "Alternative text for the image"
    },
    // scale: {
    //   type: Number,
    // },
    // target: {
    //   type: String,
    // },
    align: {
      type: String,
      // choice(["left", "center", "right", "top", "middle", "bottom"])
      doc: "The alignment of the image. Choose one of `left`, `center` or `right`"
    },
    title: {
      type: String,
      doc: "Title text for the image"
    }
  },
  run(data) {
    const { alt, height, width, align, title } = data.options || {};
    const image = {
      type: "image",
      url: data.arg,
      alt: alt !== null && alt !== void 0 ? alt : data.body ? toText(data.body) : void 0,
      title,
      height,
      width,
      align: align !== null && align !== void 0 ? align : "center"
    };
    addCommonDirectiveOptions(data, image);
    return [image];
  }
};

// ../../node_modules/myst-directives/dist/include.js
var includeDirective = {
  name: "include",
  alias: ["literalinclude"],
  doc: "Allows you to include the source or parsed version of a separate file into your document tree.",
  arg: {
    type: String,
    doc: "The file path, which is relative to the file from which it was referenced.",
    required: true
  },
  options: {
    ...commonDirectiveOptions("include"),
    literal: {
      type: Boolean,
      doc: "Flag the include block as literal, and show the contents as a code block. This can also be set automatically by setting the `language` or using the `literalinclude` directive."
    },
    lang: {
      type: String,
      doc: "The language of the code to be highlighted as. If set, this automatically changes an `include` into a `literalinclude`.",
      alias: ["language", "code"]
    },
    ...CODE_DIRECTIVE_OPTIONS,
    "start-line": {
      type: Number,
      doc: "Only the content starting from this line will be included. The first line has index 0 and negative values count from the end."
    },
    "start-at": {
      type: String,
      doc: "Only the content after and including the first occurrence of the specified text in the external data file will be included."
    },
    "start-after": {
      type: String,
      doc: "Only the content after the first occurrence of the specified text in the external data file will be included."
    },
    "end-line": {
      type: Number,
      doc: "Only the content up to (but excluding) this line will be included."
    },
    "end-at": {
      type: String,
      doc: "Only the content up to and including the first occurrence of the specified text in the external data file (but after any start-after text) will be included."
    },
    "end-before": {
      type: String,
      doc: "Only the content before the first occurrence of the specified text in the external data file (but after any start-after text) will be included."
    },
    lines: {
      type: String,
      doc: "Specify exactly which lines to include from the original file, starting at 1. For example, `1,3,5-10,20-` includes the lines 1, 3, 5 to 10 and lines 20 to the last line of the original file."
    },
    "lineno-match": {
      type: Boolean,
      doc: "Display the original line numbers, correct only when the selection consists of contiguous lines."
    }
  },
  run(data, vfile) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p;
    const literal = data.name === "literalinclude" || !!((_a = data.options) === null || _a === void 0 ? void 0 : _a.literal) || !!((_b = data.options) === null || _b === void 0 ? void 0 : _b.lang);
    const file = data.arg;
    const lang = (_d = (_c = data.options) === null || _c === void 0 ? void 0 : _c.lang) !== null && _d !== void 0 ? _d : extToLanguage(file.split(".").pop());
    const opts = literal ? getCodeBlockOptions(
      data,
      vfile,
      // Set the filename in the literal include by default
      file.split(/\/|\\/).pop()
    ) : {};
    const caption = literal ? (_e = data.options) === null || _e === void 0 ? void 0 : _e.caption : void 0;
    const filter = {};
    ensureOnlyOneOf(data, vfile, ["start-at", "start-line", "start-after", "lines"]);
    ensureOnlyOneOf(data, vfile, ["end-at", "end-line", "end-before", "lines"]);
    filter.startAt = (_f = data.options) === null || _f === void 0 ? void 0 : _f["start-at"];
    filter.startAfter = (_g = data.options) === null || _g === void 0 ? void 0 : _g["start-after"];
    filter.endAt = (_h = data.options) === null || _h === void 0 ? void 0 : _h["end-at"];
    filter.endBefore = (_j = data.options) === null || _j === void 0 ? void 0 : _j["end-before"];
    if ((_k = data.options) === null || _k === void 0 ? void 0 : _k.lines) {
      filter.lines = parseLinesString(vfile, (_l = select('mystOption[name="lines"]', data.node)) !== null && _l !== void 0 ? _l : void 0, (_m = data.options) === null || _m === void 0 ? void 0 : _m.lines);
    } else {
      const startLine = (_o = data.options) === null || _o === void 0 ? void 0 : _o["start-line"];
      const endLine = (_p = data.options) === null || _p === void 0 ? void 0 : _p["end-line"];
      const lines = [];
      if (startLine != null)
        lines.push(startLine);
      if (startLine == null && endLine != null)
        lines.push(0);
      if (endLine != null)
        lines.push(endLine);
      if (lines.length > 0) {
        filter.lines = [
          lines.map((n) => {
            if (n >= 0)
              return n + 1;
            return n;
          })
        ];
      }
    }
    const usesFilter = Object.values(filter).some((value) => value !== void 0);
    const include = {
      type: "include",
      file,
      literal,
      lang,
      caption,
      filter: usesFilter ? filter : void 0,
      ...opts
    };
    addCommonDirectiveOptions(data, include);
    return [include];
  }
};
function parseLinesString(vfile, node, linesString) {
  if (!linesString)
    return void 0;
  return linesString.split(",").map((l) => {
    const line = l.trim();
    const match = line.match(/^([0-9]+)(?:\s*(-)\s*([0-9]+)?)?$/);
    if (!match) {
      fileWarn(vfile, `Unknown lines match "${line}"`, {
        node,
        ruleId: RuleId.directiveOptionsCorrect
      });
      return void 0;
    }
    const [, first, dash, last] = match;
    if (!dash && !last) {
      return Number.parseInt(first);
    }
    if (dash && !last) {
      return [Number.parseInt(first)];
    }
    return [Number.parseInt(first), Number.parseInt(last)];
  }).filter((l) => !!l);
}
function ensureOnlyOneOf(data, vfile, exclusive) {
  const { options, node } = data;
  if (!options)
    return;
  const set1 = new Set(exclusive);
  const set2 = new Set(Object.keys(options));
  const intersection = new Set([...set1].filter((x) => set2.has(x)));
  if (intersection.size > 1) {
    fileWarn(vfile, `Conflicting options for directive: ["${[...intersection].join('", "')}"]`, {
      node,
      note: `Choose a single option out of ["${[...exclusive].join('", "')}"]`,
      ruleId: RuleId.directiveOptionsCorrect
    });
  }
}
function extToLanguage(ext) {
  var _a;
  return (_a = {
    ts: "typescript",
    js: "javascript",
    mjs: "javascript",
    tex: "latex",
    py: "python",
    md: "markdown",
    yml: "yaml"
  }[ext !== null && ext !== void 0 ? ext : ""]) !== null && _a !== void 0 ? _a : ext;
}

// ../../node_modules/myst-directives/dist/indices.js
function warnOnOptionSyntax(option, value, vfile, node) {
  fileError(vfile, `Index entry definitions should not use :option: syntax`, {
    node,
    note: `Replace ":${option}: ${value}" with "${option}: ${value}"`
  });
}
var indexDirective = {
  name: "index",
  arg: {
    type: String
  },
  options: {
    single: {
      type: String
    },
    pair: {
      type: String
    },
    triple: {
      type: String
    },
    see: {
      type: String
    },
    seealso: {
      type: String,
      alias: ["seeAlso", "see-also"]
    },
    label: {
      type: String,
      alias: ["name"]
    }
  },
  body: {
    type: String
  },
  run(data, vfile) {
    var _a, _b, _c, _d, _e, _f;
    const values = { single: [], pair: [], triple: [], see: [], seealso: [] };
    if (data.arg)
      parseIndexLine(data.arg, values, vfile, data.node);
    if ((_a = data.options) === null || _a === void 0 ? void 0 : _a.single) {
      warnOnOptionSyntax("single", data.options.single, vfile, data.node);
      values.single.push(data.options.single);
    }
    if ((_b = data.options) === null || _b === void 0 ? void 0 : _b.pair) {
      warnOnOptionSyntax("pair", data.options.pair, vfile, data.node);
      values.pair.push(data.options.pair);
    }
    if ((_c = data.options) === null || _c === void 0 ? void 0 : _c.triple) {
      warnOnOptionSyntax("triple", data.options.triple, vfile, data.node);
      values.triple.push(data.options.triple);
    }
    if ((_d = data.options) === null || _d === void 0 ? void 0 : _d.see) {
      warnOnOptionSyntax("see", data.options.see, vfile, data.node);
      values.see.push(data.options.see);
    }
    if ((_e = data.options) === null || _e === void 0 ? void 0 : _e.seealso) {
      warnOnOptionSyntax("seealso", data.options.seealso, vfile, data.node);
      values.seealso.push(data.options.seealso);
    }
    if (data.body) {
      data.body.split("\n").forEach((line) => {
        parseIndexLine(line, values, vfile, data.node);
      });
    }
    const entries = createIndexEntries(values, vfile, data.node);
    const output = [
      {
        type: "mystTarget",
        label: (_f = data.options) === null || _f === void 0 ? void 0 : _f.label,
        indexEntries: entries
      }
    ];
    return output;
  }
};
var genIndexDirective = {
  name: "show-index",
  alias: ["genindex"],
  arg: {
    type: "myst",
    doc: "Heading to be included in index block"
  },
  run(data) {
    var _a;
    const children = [];
    if (data.arg) {
      const parsedArg = data.arg;
      if (((_a = parsedArg[0]) === null || _a === void 0 ? void 0 : _a.type) === "heading") {
        children.push(...parsedArg);
      } else {
        children.push({
          type: "heading",
          depth: 2,
          enumerated: false,
          children: parsedArg
        });
      }
    }
    return [{ type: "genindex", children }];
  }
};

// ../../node_modules/csv-parse/dist/esm/sync.js
var global$1 = typeof globalThis !== "undefined" ? globalThis : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {};
var lookup = [];
var revLookup = [];
var Arr = typeof Uint8Array !== "undefined" ? Uint8Array : Array;
var inited = false;
function init() {
  inited = true;
  var code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (var i = 0, len = code.length; i < len; ++i) {
    lookup[i] = code[i];
    revLookup[code.charCodeAt(i)] = i;
  }
  revLookup["-".charCodeAt(0)] = 62;
  revLookup["_".charCodeAt(0)] = 63;
}
function toByteArray(b64) {
  if (!inited) {
    init();
  }
  var i, j, l, tmp, placeHolders, arr;
  var len = b64.length;
  if (len % 4 > 0) {
    throw new Error("Invalid string. Length must be a multiple of 4");
  }
  placeHolders = b64[len - 2] === "=" ? 2 : b64[len - 1] === "=" ? 1 : 0;
  arr = new Arr(len * 3 / 4 - placeHolders);
  l = placeHolders > 0 ? len - 4 : len;
  var L = 0;
  for (i = 0, j = 0; i < l; i += 4, j += 3) {
    tmp = revLookup[b64.charCodeAt(i)] << 18 | revLookup[b64.charCodeAt(i + 1)] << 12 | revLookup[b64.charCodeAt(i + 2)] << 6 | revLookup[b64.charCodeAt(i + 3)];
    arr[L++] = tmp >> 16 & 255;
    arr[L++] = tmp >> 8 & 255;
    arr[L++] = tmp & 255;
  }
  if (placeHolders === 2) {
    tmp = revLookup[b64.charCodeAt(i)] << 2 | revLookup[b64.charCodeAt(i + 1)] >> 4;
    arr[L++] = tmp & 255;
  } else if (placeHolders === 1) {
    tmp = revLookup[b64.charCodeAt(i)] << 10 | revLookup[b64.charCodeAt(i + 1)] << 4 | revLookup[b64.charCodeAt(i + 2)] >> 2;
    arr[L++] = tmp >> 8 & 255;
    arr[L++] = tmp & 255;
  }
  return arr;
}
function tripletToBase64(num) {
  return lookup[num >> 18 & 63] + lookup[num >> 12 & 63] + lookup[num >> 6 & 63] + lookup[num & 63];
}
function encodeChunk(uint8, start, end) {
  var tmp;
  var output = [];
  for (var i = start; i < end; i += 3) {
    tmp = (uint8[i] << 16) + (uint8[i + 1] << 8) + uint8[i + 2];
    output.push(tripletToBase64(tmp));
  }
  return output.join("");
}
function fromByteArray(uint8) {
  if (!inited) {
    init();
  }
  var tmp;
  var len = uint8.length;
  var extraBytes = len % 3;
  var output = "";
  var parts = [];
  var maxChunkLength = 16383;
  for (var i = 0, len2 = len - extraBytes; i < len2; i += maxChunkLength) {
    parts.push(encodeChunk(uint8, i, i + maxChunkLength > len2 ? len2 : i + maxChunkLength));
  }
  if (extraBytes === 1) {
    tmp = uint8[len - 1];
    output += lookup[tmp >> 2];
    output += lookup[tmp << 4 & 63];
    output += "==";
  } else if (extraBytes === 2) {
    tmp = (uint8[len - 2] << 8) + uint8[len - 1];
    output += lookup[tmp >> 10];
    output += lookup[tmp >> 4 & 63];
    output += lookup[tmp << 2 & 63];
    output += "=";
  }
  parts.push(output);
  return parts.join("");
}
function read(buffer, offset, isLE, mLen, nBytes) {
  var e, m;
  var eLen = nBytes * 8 - mLen - 1;
  var eMax = (1 << eLen) - 1;
  var eBias = eMax >> 1;
  var nBits = -7;
  var i = isLE ? nBytes - 1 : 0;
  var d = isLE ? -1 : 1;
  var s = buffer[offset + i];
  i += d;
  e = s & (1 << -nBits) - 1;
  s >>= -nBits;
  nBits += eLen;
  for (; nBits > 0; e = e * 256 + buffer[offset + i], i += d, nBits -= 8) {
  }
  m = e & (1 << -nBits) - 1;
  e >>= -nBits;
  nBits += mLen;
  for (; nBits > 0; m = m * 256 + buffer[offset + i], i += d, nBits -= 8) {
  }
  if (e === 0) {
    e = 1 - eBias;
  } else if (e === eMax) {
    return m ? NaN : (s ? -1 : 1) * Infinity;
  } else {
    m = m + Math.pow(2, mLen);
    e = e - eBias;
  }
  return (s ? -1 : 1) * m * Math.pow(2, e - mLen);
}
function write(buffer, value, offset, isLE, mLen, nBytes) {
  var e, m, c;
  var eLen = nBytes * 8 - mLen - 1;
  var eMax = (1 << eLen) - 1;
  var eBias = eMax >> 1;
  var rt = mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0;
  var i = isLE ? 0 : nBytes - 1;
  var d = isLE ? 1 : -1;
  var s = value < 0 || value === 0 && 1 / value < 0 ? 1 : 0;
  value = Math.abs(value);
  if (isNaN(value) || value === Infinity) {
    m = isNaN(value) ? 1 : 0;
    e = eMax;
  } else {
    e = Math.floor(Math.log(value) / Math.LN2);
    if (value * (c = Math.pow(2, -e)) < 1) {
      e--;
      c *= 2;
    }
    if (e + eBias >= 1) {
      value += rt / c;
    } else {
      value += rt * Math.pow(2, 1 - eBias);
    }
    if (value * c >= 2) {
      e++;
      c /= 2;
    }
    if (e + eBias >= eMax) {
      m = 0;
      e = eMax;
    } else if (e + eBias >= 1) {
      m = (value * c - 1) * Math.pow(2, mLen);
      e = e + eBias;
    } else {
      m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
      e = 0;
    }
  }
  for (; mLen >= 8; buffer[offset + i] = m & 255, i += d, m /= 256, mLen -= 8) {
  }
  e = e << mLen | m;
  eLen += mLen;
  for (; eLen > 0; buffer[offset + i] = e & 255, i += d, e /= 256, eLen -= 8) {
  }
  buffer[offset + i - d] |= s * 128;
}
var toString = {}.toString;
var isArray = Array.isArray || function(arr) {
  return toString.call(arr) == "[object Array]";
};
var INSPECT_MAX_BYTES = 50;
Buffer.TYPED_ARRAY_SUPPORT = global$1.TYPED_ARRAY_SUPPORT !== void 0 ? global$1.TYPED_ARRAY_SUPPORT : true;
kMaxLength();
function kMaxLength() {
  return Buffer.TYPED_ARRAY_SUPPORT ? 2147483647 : 1073741823;
}
function createBuffer(that, length) {
  if (kMaxLength() < length) {
    throw new RangeError("Invalid typed array length");
  }
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    that = new Uint8Array(length);
    that.__proto__ = Buffer.prototype;
  } else {
    if (that === null) {
      that = new Buffer(length);
    }
    that.length = length;
  }
  return that;
}
function Buffer(arg, encodingOrOffset, length) {
  if (!Buffer.TYPED_ARRAY_SUPPORT && !(this instanceof Buffer)) {
    return new Buffer(arg, encodingOrOffset, length);
  }
  if (typeof arg === "number") {
    if (typeof encodingOrOffset === "string") {
      throw new Error(
        "If encoding is specified then the first argument must be a string"
      );
    }
    return allocUnsafe(this, arg);
  }
  return from(this, arg, encodingOrOffset, length);
}
Buffer.poolSize = 8192;
Buffer._augment = function(arr) {
  arr.__proto__ = Buffer.prototype;
  return arr;
};
function from(that, value, encodingOrOffset, length) {
  if (typeof value === "number") {
    throw new TypeError('"value" argument must not be a number');
  }
  if (typeof ArrayBuffer !== "undefined" && value instanceof ArrayBuffer) {
    return fromArrayBuffer(that, value, encodingOrOffset, length);
  }
  if (typeof value === "string") {
    return fromString(that, value, encodingOrOffset);
  }
  return fromObject(that, value);
}
Buffer.from = function(value, encodingOrOffset, length) {
  return from(null, value, encodingOrOffset, length);
};
if (Buffer.TYPED_ARRAY_SUPPORT) {
  Buffer.prototype.__proto__ = Uint8Array.prototype;
  Buffer.__proto__ = Uint8Array;
  if (typeof Symbol !== "undefined" && Symbol.species && Buffer[Symbol.species] === Buffer)
    ;
}
function assertSize(size) {
  if (typeof size !== "number") {
    throw new TypeError('"size" argument must be a number');
  } else if (size < 0) {
    throw new RangeError('"size" argument must not be negative');
  }
}
function alloc(that, size, fill2, encoding) {
  assertSize(size);
  if (size <= 0) {
    return createBuffer(that, size);
  }
  if (fill2 !== void 0) {
    return typeof encoding === "string" ? createBuffer(that, size).fill(fill2, encoding) : createBuffer(that, size).fill(fill2);
  }
  return createBuffer(that, size);
}
Buffer.alloc = function(size, fill2, encoding) {
  return alloc(null, size, fill2, encoding);
};
function allocUnsafe(that, size) {
  assertSize(size);
  that = createBuffer(that, size < 0 ? 0 : checked(size) | 0);
  if (!Buffer.TYPED_ARRAY_SUPPORT) {
    for (var i = 0; i < size; ++i) {
      that[i] = 0;
    }
  }
  return that;
}
Buffer.allocUnsafe = function(size) {
  return allocUnsafe(null, size);
};
Buffer.allocUnsafeSlow = function(size) {
  return allocUnsafe(null, size);
};
function fromString(that, string, encoding) {
  if (typeof encoding !== "string" || encoding === "") {
    encoding = "utf8";
  }
  if (!Buffer.isEncoding(encoding)) {
    throw new TypeError('"encoding" must be a valid string encoding');
  }
  var length = byteLength(string, encoding) | 0;
  that = createBuffer(that, length);
  var actual = that.write(string, encoding);
  if (actual !== length) {
    that = that.slice(0, actual);
  }
  return that;
}
function fromArrayLike(that, array) {
  var length = array.length < 0 ? 0 : checked(array.length) | 0;
  that = createBuffer(that, length);
  for (var i = 0; i < length; i += 1) {
    that[i] = array[i] & 255;
  }
  return that;
}
function fromArrayBuffer(that, array, byteOffset, length) {
  array.byteLength;
  if (byteOffset < 0 || array.byteLength < byteOffset) {
    throw new RangeError("'offset' is out of bounds");
  }
  if (array.byteLength < byteOffset + (length || 0)) {
    throw new RangeError("'length' is out of bounds");
  }
  if (byteOffset === void 0 && length === void 0) {
    array = new Uint8Array(array);
  } else if (length === void 0) {
    array = new Uint8Array(array, byteOffset);
  } else {
    array = new Uint8Array(array, byteOffset, length);
  }
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    that = array;
    that.__proto__ = Buffer.prototype;
  } else {
    that = fromArrayLike(that, array);
  }
  return that;
}
function fromObject(that, obj) {
  if (internalIsBuffer(obj)) {
    var len = checked(obj.length) | 0;
    that = createBuffer(that, len);
    if (that.length === 0) {
      return that;
    }
    obj.copy(that, 0, 0, len);
    return that;
  }
  if (obj) {
    if (typeof ArrayBuffer !== "undefined" && obj.buffer instanceof ArrayBuffer || "length" in obj) {
      if (typeof obj.length !== "number" || isnan(obj.length)) {
        return createBuffer(that, 0);
      }
      return fromArrayLike(that, obj);
    }
    if (obj.type === "Buffer" && isArray(obj.data)) {
      return fromArrayLike(that, obj.data);
    }
  }
  throw new TypeError("First argument must be a string, Buffer, ArrayBuffer, Array, or array-like object.");
}
function checked(length) {
  if (length >= kMaxLength()) {
    throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x" + kMaxLength().toString(16) + " bytes");
  }
  return length | 0;
}
Buffer.isBuffer = isBuffer;
function internalIsBuffer(b) {
  return !!(b != null && b._isBuffer);
}
Buffer.compare = function compare(a, b) {
  if (!internalIsBuffer(a) || !internalIsBuffer(b)) {
    throw new TypeError("Arguments must be Buffers");
  }
  if (a === b)
    return 0;
  var x = a.length;
  var y = b.length;
  for (var i = 0, len = Math.min(x, y); i < len; ++i) {
    if (a[i] !== b[i]) {
      x = a[i];
      y = b[i];
      break;
    }
  }
  if (x < y)
    return -1;
  if (y < x)
    return 1;
  return 0;
};
Buffer.isEncoding = function isEncoding(encoding) {
  switch (String(encoding).toLowerCase()) {
    case "hex":
    case "utf8":
    case "utf-8":
    case "ascii":
    case "latin1":
    case "binary":
    case "base64":
    case "ucs2":
    case "ucs-2":
    case "utf16le":
    case "utf-16le":
      return true;
    default:
      return false;
  }
};
Buffer.concat = function concat(list, length) {
  if (!isArray(list)) {
    throw new TypeError('"list" argument must be an Array of Buffers');
  }
  if (list.length === 0) {
    return Buffer.alloc(0);
  }
  var i;
  if (length === void 0) {
    length = 0;
    for (i = 0; i < list.length; ++i) {
      length += list[i].length;
    }
  }
  var buffer = Buffer.allocUnsafe(length);
  var pos = 0;
  for (i = 0; i < list.length; ++i) {
    var buf = list[i];
    if (!internalIsBuffer(buf)) {
      throw new TypeError('"list" argument must be an Array of Buffers');
    }
    buf.copy(buffer, pos);
    pos += buf.length;
  }
  return buffer;
};
function byteLength(string, encoding) {
  if (internalIsBuffer(string)) {
    return string.length;
  }
  if (typeof ArrayBuffer !== "undefined" && typeof ArrayBuffer.isView === "function" && (ArrayBuffer.isView(string) || string instanceof ArrayBuffer)) {
    return string.byteLength;
  }
  if (typeof string !== "string") {
    string = "" + string;
  }
  var len = string.length;
  if (len === 0)
    return 0;
  var loweredCase = false;
  for (; ; ) {
    switch (encoding) {
      case "ascii":
      case "latin1":
      case "binary":
        return len;
      case "utf8":
      case "utf-8":
      case void 0:
        return utf8ToBytes(string).length;
      case "ucs2":
      case "ucs-2":
      case "utf16le":
      case "utf-16le":
        return len * 2;
      case "hex":
        return len >>> 1;
      case "base64":
        return base64ToBytes(string).length;
      default:
        if (loweredCase)
          return utf8ToBytes(string).length;
        encoding = ("" + encoding).toLowerCase();
        loweredCase = true;
    }
  }
}
Buffer.byteLength = byteLength;
function slowToString(encoding, start, end) {
  var loweredCase = false;
  if (start === void 0 || start < 0) {
    start = 0;
  }
  if (start > this.length) {
    return "";
  }
  if (end === void 0 || end > this.length) {
    end = this.length;
  }
  if (end <= 0) {
    return "";
  }
  end >>>= 0;
  start >>>= 0;
  if (end <= start) {
    return "";
  }
  if (!encoding)
    encoding = "utf8";
  while (true) {
    switch (encoding) {
      case "hex":
        return hexSlice(this, start, end);
      case "utf8":
      case "utf-8":
        return utf8Slice(this, start, end);
      case "ascii":
        return asciiSlice(this, start, end);
      case "latin1":
      case "binary":
        return latin1Slice(this, start, end);
      case "base64":
        return base64Slice(this, start, end);
      case "ucs2":
      case "ucs-2":
      case "utf16le":
      case "utf-16le":
        return utf16leSlice(this, start, end);
      default:
        if (loweredCase)
          throw new TypeError("Unknown encoding: " + encoding);
        encoding = (encoding + "").toLowerCase();
        loweredCase = true;
    }
  }
}
Buffer.prototype._isBuffer = true;
function swap(b, n, m) {
  var i = b[n];
  b[n] = b[m];
  b[m] = i;
}
Buffer.prototype.swap16 = function swap16() {
  var len = this.length;
  if (len % 2 !== 0) {
    throw new RangeError("Buffer size must be a multiple of 16-bits");
  }
  for (var i = 0; i < len; i += 2) {
    swap(this, i, i + 1);
  }
  return this;
};
Buffer.prototype.swap32 = function swap32() {
  var len = this.length;
  if (len % 4 !== 0) {
    throw new RangeError("Buffer size must be a multiple of 32-bits");
  }
  for (var i = 0; i < len; i += 4) {
    swap(this, i, i + 3);
    swap(this, i + 1, i + 2);
  }
  return this;
};
Buffer.prototype.swap64 = function swap64() {
  var len = this.length;
  if (len % 8 !== 0) {
    throw new RangeError("Buffer size must be a multiple of 64-bits");
  }
  for (var i = 0; i < len; i += 8) {
    swap(this, i, i + 7);
    swap(this, i + 1, i + 6);
    swap(this, i + 2, i + 5);
    swap(this, i + 3, i + 4);
  }
  return this;
};
Buffer.prototype.toString = function toString2() {
  var length = this.length | 0;
  if (length === 0)
    return "";
  if (arguments.length === 0)
    return utf8Slice(this, 0, length);
  return slowToString.apply(this, arguments);
};
Buffer.prototype.equals = function equals(b) {
  if (!internalIsBuffer(b))
    throw new TypeError("Argument must be a Buffer");
  if (this === b)
    return true;
  return Buffer.compare(this, b) === 0;
};
Buffer.prototype.inspect = function inspect() {
  var str = "";
  var max = INSPECT_MAX_BYTES;
  if (this.length > 0) {
    str = this.toString("hex", 0, max).match(/.{2}/g).join(" ");
    if (this.length > max)
      str += " ... ";
  }
  return "<Buffer " + str + ">";
};
Buffer.prototype.compare = function compare2(target, start, end, thisStart, thisEnd) {
  if (!internalIsBuffer(target)) {
    throw new TypeError("Argument must be a Buffer");
  }
  if (start === void 0) {
    start = 0;
  }
  if (end === void 0) {
    end = target ? target.length : 0;
  }
  if (thisStart === void 0) {
    thisStart = 0;
  }
  if (thisEnd === void 0) {
    thisEnd = this.length;
  }
  if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
    throw new RangeError("out of range index");
  }
  if (thisStart >= thisEnd && start >= end) {
    return 0;
  }
  if (thisStart >= thisEnd) {
    return -1;
  }
  if (start >= end) {
    return 1;
  }
  start >>>= 0;
  end >>>= 0;
  thisStart >>>= 0;
  thisEnd >>>= 0;
  if (this === target)
    return 0;
  var x = thisEnd - thisStart;
  var y = end - start;
  var len = Math.min(x, y);
  var thisCopy = this.slice(thisStart, thisEnd);
  var targetCopy = target.slice(start, end);
  for (var i = 0; i < len; ++i) {
    if (thisCopy[i] !== targetCopy[i]) {
      x = thisCopy[i];
      y = targetCopy[i];
      break;
    }
  }
  if (x < y)
    return -1;
  if (y < x)
    return 1;
  return 0;
};
function bidirectionalIndexOf(buffer, val, byteOffset, encoding, dir) {
  if (buffer.length === 0)
    return -1;
  if (typeof byteOffset === "string") {
    encoding = byteOffset;
    byteOffset = 0;
  } else if (byteOffset > 2147483647) {
    byteOffset = 2147483647;
  } else if (byteOffset < -2147483648) {
    byteOffset = -2147483648;
  }
  byteOffset = +byteOffset;
  if (isNaN(byteOffset)) {
    byteOffset = dir ? 0 : buffer.length - 1;
  }
  if (byteOffset < 0)
    byteOffset = buffer.length + byteOffset;
  if (byteOffset >= buffer.length) {
    if (dir)
      return -1;
    else
      byteOffset = buffer.length - 1;
  } else if (byteOffset < 0) {
    if (dir)
      byteOffset = 0;
    else
      return -1;
  }
  if (typeof val === "string") {
    val = Buffer.from(val, encoding);
  }
  if (internalIsBuffer(val)) {
    if (val.length === 0) {
      return -1;
    }
    return arrayIndexOf(buffer, val, byteOffset, encoding, dir);
  } else if (typeof val === "number") {
    val = val & 255;
    if (Buffer.TYPED_ARRAY_SUPPORT && typeof Uint8Array.prototype.indexOf === "function") {
      if (dir) {
        return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset);
      } else {
        return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset);
      }
    }
    return arrayIndexOf(buffer, [val], byteOffset, encoding, dir);
  }
  throw new TypeError("val must be string, number or Buffer");
}
function arrayIndexOf(arr, val, byteOffset, encoding, dir) {
  var indexSize = 1;
  var arrLength = arr.length;
  var valLength = val.length;
  if (encoding !== void 0) {
    encoding = String(encoding).toLowerCase();
    if (encoding === "ucs2" || encoding === "ucs-2" || encoding === "utf16le" || encoding === "utf-16le") {
      if (arr.length < 2 || val.length < 2) {
        return -1;
      }
      indexSize = 2;
      arrLength /= 2;
      valLength /= 2;
      byteOffset /= 2;
    }
  }
  function read2(buf, i2) {
    if (indexSize === 1) {
      return buf[i2];
    } else {
      return buf.readUInt16BE(i2 * indexSize);
    }
  }
  var i;
  if (dir) {
    var foundIndex = -1;
    for (i = byteOffset; i < arrLength; i++) {
      if (read2(arr, i) === read2(val, foundIndex === -1 ? 0 : i - foundIndex)) {
        if (foundIndex === -1)
          foundIndex = i;
        if (i - foundIndex + 1 === valLength)
          return foundIndex * indexSize;
      } else {
        if (foundIndex !== -1)
          i -= i - foundIndex;
        foundIndex = -1;
      }
    }
  } else {
    if (byteOffset + valLength > arrLength)
      byteOffset = arrLength - valLength;
    for (i = byteOffset; i >= 0; i--) {
      var found = true;
      for (var j = 0; j < valLength; j++) {
        if (read2(arr, i + j) !== read2(val, j)) {
          found = false;
          break;
        }
      }
      if (found)
        return i;
    }
  }
  return -1;
}
Buffer.prototype.includes = function includes(val, byteOffset, encoding) {
  return this.indexOf(val, byteOffset, encoding) !== -1;
};
Buffer.prototype.indexOf = function indexOf(val, byteOffset, encoding) {
  return bidirectionalIndexOf(this, val, byteOffset, encoding, true);
};
Buffer.prototype.lastIndexOf = function lastIndexOf(val, byteOffset, encoding) {
  return bidirectionalIndexOf(this, val, byteOffset, encoding, false);
};
function hexWrite(buf, string, offset, length) {
  offset = Number(offset) || 0;
  var remaining = buf.length - offset;
  if (!length) {
    length = remaining;
  } else {
    length = Number(length);
    if (length > remaining) {
      length = remaining;
    }
  }
  var strLen = string.length;
  if (strLen % 2 !== 0)
    throw new TypeError("Invalid hex string");
  if (length > strLen / 2) {
    length = strLen / 2;
  }
  for (var i = 0; i < length; ++i) {
    var parsed = parseInt(string.substr(i * 2, 2), 16);
    if (isNaN(parsed))
      return i;
    buf[offset + i] = parsed;
  }
  return i;
}
function utf8Write(buf, string, offset, length) {
  return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length);
}
function asciiWrite(buf, string, offset, length) {
  return blitBuffer(asciiToBytes(string), buf, offset, length);
}
function latin1Write(buf, string, offset, length) {
  return asciiWrite(buf, string, offset, length);
}
function base64Write(buf, string, offset, length) {
  return blitBuffer(base64ToBytes(string), buf, offset, length);
}
function ucs2Write(buf, string, offset, length) {
  return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length);
}
Buffer.prototype.write = function write2(string, offset, length, encoding) {
  if (offset === void 0) {
    encoding = "utf8";
    length = this.length;
    offset = 0;
  } else if (length === void 0 && typeof offset === "string") {
    encoding = offset;
    length = this.length;
    offset = 0;
  } else if (isFinite(offset)) {
    offset = offset | 0;
    if (isFinite(length)) {
      length = length | 0;
      if (encoding === void 0)
        encoding = "utf8";
    } else {
      encoding = length;
      length = void 0;
    }
  } else {
    throw new Error(
      "Buffer.write(string, encoding, offset[, length]) is no longer supported"
    );
  }
  var remaining = this.length - offset;
  if (length === void 0 || length > remaining)
    length = remaining;
  if (string.length > 0 && (length < 0 || offset < 0) || offset > this.length) {
    throw new RangeError("Attempt to write outside buffer bounds");
  }
  if (!encoding)
    encoding = "utf8";
  var loweredCase = false;
  for (; ; ) {
    switch (encoding) {
      case "hex":
        return hexWrite(this, string, offset, length);
      case "utf8":
      case "utf-8":
        return utf8Write(this, string, offset, length);
      case "ascii":
        return asciiWrite(this, string, offset, length);
      case "latin1":
      case "binary":
        return latin1Write(this, string, offset, length);
      case "base64":
        return base64Write(this, string, offset, length);
      case "ucs2":
      case "ucs-2":
      case "utf16le":
      case "utf-16le":
        return ucs2Write(this, string, offset, length);
      default:
        if (loweredCase)
          throw new TypeError("Unknown encoding: " + encoding);
        encoding = ("" + encoding).toLowerCase();
        loweredCase = true;
    }
  }
};
Buffer.prototype.toJSON = function toJSON() {
  return {
    type: "Buffer",
    data: Array.prototype.slice.call(this._arr || this, 0)
  };
};
function base64Slice(buf, start, end) {
  if (start === 0 && end === buf.length) {
    return fromByteArray(buf);
  } else {
    return fromByteArray(buf.slice(start, end));
  }
}
function utf8Slice(buf, start, end) {
  end = Math.min(buf.length, end);
  var res = [];
  var i = start;
  while (i < end) {
    var firstByte = buf[i];
    var codePoint = null;
    var bytesPerSequence = firstByte > 239 ? 4 : firstByte > 223 ? 3 : firstByte > 191 ? 2 : 1;
    if (i + bytesPerSequence <= end) {
      var secondByte, thirdByte, fourthByte, tempCodePoint;
      switch (bytesPerSequence) {
        case 1:
          if (firstByte < 128) {
            codePoint = firstByte;
          }
          break;
        case 2:
          secondByte = buf[i + 1];
          if ((secondByte & 192) === 128) {
            tempCodePoint = (firstByte & 31) << 6 | secondByte & 63;
            if (tempCodePoint > 127) {
              codePoint = tempCodePoint;
            }
          }
          break;
        case 3:
          secondByte = buf[i + 1];
          thirdByte = buf[i + 2];
          if ((secondByte & 192) === 128 && (thirdByte & 192) === 128) {
            tempCodePoint = (firstByte & 15) << 12 | (secondByte & 63) << 6 | thirdByte & 63;
            if (tempCodePoint > 2047 && (tempCodePoint < 55296 || tempCodePoint > 57343)) {
              codePoint = tempCodePoint;
            }
          }
          break;
        case 4:
          secondByte = buf[i + 1];
          thirdByte = buf[i + 2];
          fourthByte = buf[i + 3];
          if ((secondByte & 192) === 128 && (thirdByte & 192) === 128 && (fourthByte & 192) === 128) {
            tempCodePoint = (firstByte & 15) << 18 | (secondByte & 63) << 12 | (thirdByte & 63) << 6 | fourthByte & 63;
            if (tempCodePoint > 65535 && tempCodePoint < 1114112) {
              codePoint = tempCodePoint;
            }
          }
      }
    }
    if (codePoint === null) {
      codePoint = 65533;
      bytesPerSequence = 1;
    } else if (codePoint > 65535) {
      codePoint -= 65536;
      res.push(codePoint >>> 10 & 1023 | 55296);
      codePoint = 56320 | codePoint & 1023;
    }
    res.push(codePoint);
    i += bytesPerSequence;
  }
  return decodeCodePointsArray(res);
}
var MAX_ARGUMENTS_LENGTH = 4096;
function decodeCodePointsArray(codePoints) {
  var len = codePoints.length;
  if (len <= MAX_ARGUMENTS_LENGTH) {
    return String.fromCharCode.apply(String, codePoints);
  }
  var res = "";
  var i = 0;
  while (i < len) {
    res += String.fromCharCode.apply(
      String,
      codePoints.slice(i, i += MAX_ARGUMENTS_LENGTH)
    );
  }
  return res;
}
function asciiSlice(buf, start, end) {
  var ret = "";
  end = Math.min(buf.length, end);
  for (var i = start; i < end; ++i) {
    ret += String.fromCharCode(buf[i] & 127);
  }
  return ret;
}
function latin1Slice(buf, start, end) {
  var ret = "";
  end = Math.min(buf.length, end);
  for (var i = start; i < end; ++i) {
    ret += String.fromCharCode(buf[i]);
  }
  return ret;
}
function hexSlice(buf, start, end) {
  var len = buf.length;
  if (!start || start < 0)
    start = 0;
  if (!end || end < 0 || end > len)
    end = len;
  var out = "";
  for (var i = start; i < end; ++i) {
    out += toHex(buf[i]);
  }
  return out;
}
function utf16leSlice(buf, start, end) {
  var bytes = buf.slice(start, end);
  var res = "";
  for (var i = 0; i < bytes.length; i += 2) {
    res += String.fromCharCode(bytes[i] + bytes[i + 1] * 256);
  }
  return res;
}
Buffer.prototype.slice = function slice(start, end) {
  var len = this.length;
  start = ~~start;
  end = end === void 0 ? len : ~~end;
  if (start < 0) {
    start += len;
    if (start < 0)
      start = 0;
  } else if (start > len) {
    start = len;
  }
  if (end < 0) {
    end += len;
    if (end < 0)
      end = 0;
  } else if (end > len) {
    end = len;
  }
  if (end < start)
    end = start;
  var newBuf;
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    newBuf = this.subarray(start, end);
    newBuf.__proto__ = Buffer.prototype;
  } else {
    var sliceLen = end - start;
    newBuf = new Buffer(sliceLen, void 0);
    for (var i = 0; i < sliceLen; ++i) {
      newBuf[i] = this[i + start];
    }
  }
  return newBuf;
};
function checkOffset(offset, ext, length) {
  if (offset % 1 !== 0 || offset < 0)
    throw new RangeError("offset is not uint");
  if (offset + ext > length)
    throw new RangeError("Trying to access beyond buffer length");
}
Buffer.prototype.readUIntLE = function readUIntLE(offset, byteLength2, noAssert) {
  offset = offset | 0;
  byteLength2 = byteLength2 | 0;
  if (!noAssert)
    checkOffset(offset, byteLength2, this.length);
  var val = this[offset];
  var mul = 1;
  var i = 0;
  while (++i < byteLength2 && (mul *= 256)) {
    val += this[offset + i] * mul;
  }
  return val;
};
Buffer.prototype.readUIntBE = function readUIntBE(offset, byteLength2, noAssert) {
  offset = offset | 0;
  byteLength2 = byteLength2 | 0;
  if (!noAssert) {
    checkOffset(offset, byteLength2, this.length);
  }
  var val = this[offset + --byteLength2];
  var mul = 1;
  while (byteLength2 > 0 && (mul *= 256)) {
    val += this[offset + --byteLength2] * mul;
  }
  return val;
};
Buffer.prototype.readUInt8 = function readUInt8(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 1, this.length);
  return this[offset];
};
Buffer.prototype.readUInt16LE = function readUInt16LE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 2, this.length);
  return this[offset] | this[offset + 1] << 8;
};
Buffer.prototype.readUInt16BE = function readUInt16BE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 2, this.length);
  return this[offset] << 8 | this[offset + 1];
};
Buffer.prototype.readUInt32LE = function readUInt32LE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 4, this.length);
  return (this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16) + this[offset + 3] * 16777216;
};
Buffer.prototype.readUInt32BE = function readUInt32BE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 4, this.length);
  return this[offset] * 16777216 + (this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3]);
};
Buffer.prototype.readIntLE = function readIntLE(offset, byteLength2, noAssert) {
  offset = offset | 0;
  byteLength2 = byteLength2 | 0;
  if (!noAssert)
    checkOffset(offset, byteLength2, this.length);
  var val = this[offset];
  var mul = 1;
  var i = 0;
  while (++i < byteLength2 && (mul *= 256)) {
    val += this[offset + i] * mul;
  }
  mul *= 128;
  if (val >= mul)
    val -= Math.pow(2, 8 * byteLength2);
  return val;
};
Buffer.prototype.readIntBE = function readIntBE(offset, byteLength2, noAssert) {
  offset = offset | 0;
  byteLength2 = byteLength2 | 0;
  if (!noAssert)
    checkOffset(offset, byteLength2, this.length);
  var i = byteLength2;
  var mul = 1;
  var val = this[offset + --i];
  while (i > 0 && (mul *= 256)) {
    val += this[offset + --i] * mul;
  }
  mul *= 128;
  if (val >= mul)
    val -= Math.pow(2, 8 * byteLength2);
  return val;
};
Buffer.prototype.readInt8 = function readInt8(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 1, this.length);
  if (!(this[offset] & 128))
    return this[offset];
  return (255 - this[offset] + 1) * -1;
};
Buffer.prototype.readInt16LE = function readInt16LE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 2, this.length);
  var val = this[offset] | this[offset + 1] << 8;
  return val & 32768 ? val | 4294901760 : val;
};
Buffer.prototype.readInt16BE = function readInt16BE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 2, this.length);
  var val = this[offset + 1] | this[offset] << 8;
  return val & 32768 ? val | 4294901760 : val;
};
Buffer.prototype.readInt32LE = function readInt32LE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 4, this.length);
  return this[offset] | this[offset + 1] << 8 | this[offset + 2] << 16 | this[offset + 3] << 24;
};
Buffer.prototype.readInt32BE = function readInt32BE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 4, this.length);
  return this[offset] << 24 | this[offset + 1] << 16 | this[offset + 2] << 8 | this[offset + 3];
};
Buffer.prototype.readFloatLE = function readFloatLE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 4, this.length);
  return read(this, offset, true, 23, 4);
};
Buffer.prototype.readFloatBE = function readFloatBE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 4, this.length);
  return read(this, offset, false, 23, 4);
};
Buffer.prototype.readDoubleLE = function readDoubleLE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 8, this.length);
  return read(this, offset, true, 52, 8);
};
Buffer.prototype.readDoubleBE = function readDoubleBE(offset, noAssert) {
  if (!noAssert)
    checkOffset(offset, 8, this.length);
  return read(this, offset, false, 52, 8);
};
function checkInt(buf, value, offset, ext, max, min) {
  if (!internalIsBuffer(buf))
    throw new TypeError('"buffer" argument must be a Buffer instance');
  if (value > max || value < min)
    throw new RangeError('"value" argument is out of bounds');
  if (offset + ext > buf.length)
    throw new RangeError("Index out of range");
}
Buffer.prototype.writeUIntLE = function writeUIntLE(value, offset, byteLength2, noAssert) {
  value = +value;
  offset = offset | 0;
  byteLength2 = byteLength2 | 0;
  if (!noAssert) {
    var maxBytes = Math.pow(2, 8 * byteLength2) - 1;
    checkInt(this, value, offset, byteLength2, maxBytes, 0);
  }
  var mul = 1;
  var i = 0;
  this[offset] = value & 255;
  while (++i < byteLength2 && (mul *= 256)) {
    this[offset + i] = value / mul & 255;
  }
  return offset + byteLength2;
};
Buffer.prototype.writeUIntBE = function writeUIntBE(value, offset, byteLength2, noAssert) {
  value = +value;
  offset = offset | 0;
  byteLength2 = byteLength2 | 0;
  if (!noAssert) {
    var maxBytes = Math.pow(2, 8 * byteLength2) - 1;
    checkInt(this, value, offset, byteLength2, maxBytes, 0);
  }
  var i = byteLength2 - 1;
  var mul = 1;
  this[offset + i] = value & 255;
  while (--i >= 0 && (mul *= 256)) {
    this[offset + i] = value / mul & 255;
  }
  return offset + byteLength2;
};
Buffer.prototype.writeUInt8 = function writeUInt8(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 1, 255, 0);
  if (!Buffer.TYPED_ARRAY_SUPPORT)
    value = Math.floor(value);
  this[offset] = value & 255;
  return offset + 1;
};
function objectWriteUInt16(buf, value, offset, littleEndian) {
  if (value < 0)
    value = 65535 + value + 1;
  for (var i = 0, j = Math.min(buf.length - offset, 2); i < j; ++i) {
    buf[offset + i] = (value & 255 << 8 * (littleEndian ? i : 1 - i)) >>> (littleEndian ? i : 1 - i) * 8;
  }
}
Buffer.prototype.writeUInt16LE = function writeUInt16LE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 2, 65535, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
  } else {
    objectWriteUInt16(this, value, offset, true);
  }
  return offset + 2;
};
Buffer.prototype.writeUInt16BE = function writeUInt16BE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 2, 65535, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value >>> 8;
    this[offset + 1] = value & 255;
  } else {
    objectWriteUInt16(this, value, offset, false);
  }
  return offset + 2;
};
function objectWriteUInt32(buf, value, offset, littleEndian) {
  if (value < 0)
    value = 4294967295 + value + 1;
  for (var i = 0, j = Math.min(buf.length - offset, 4); i < j; ++i) {
    buf[offset + i] = value >>> (littleEndian ? i : 3 - i) * 8 & 255;
  }
}
Buffer.prototype.writeUInt32LE = function writeUInt32LE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 4, 4294967295, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset + 3] = value >>> 24;
    this[offset + 2] = value >>> 16;
    this[offset + 1] = value >>> 8;
    this[offset] = value & 255;
  } else {
    objectWriteUInt32(this, value, offset, true);
  }
  return offset + 4;
};
Buffer.prototype.writeUInt32BE = function writeUInt32BE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 4, 4294967295, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value >>> 24;
    this[offset + 1] = value >>> 16;
    this[offset + 2] = value >>> 8;
    this[offset + 3] = value & 255;
  } else {
    objectWriteUInt32(this, value, offset, false);
  }
  return offset + 4;
};
Buffer.prototype.writeIntLE = function writeIntLE(value, offset, byteLength2, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) {
    var limit = Math.pow(2, 8 * byteLength2 - 1);
    checkInt(this, value, offset, byteLength2, limit - 1, -limit);
  }
  var i = 0;
  var mul = 1;
  var sub = 0;
  this[offset] = value & 255;
  while (++i < byteLength2 && (mul *= 256)) {
    if (value < 0 && sub === 0 && this[offset + i - 1] !== 0) {
      sub = 1;
    }
    this[offset + i] = (value / mul >> 0) - sub & 255;
  }
  return offset + byteLength2;
};
Buffer.prototype.writeIntBE = function writeIntBE(value, offset, byteLength2, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) {
    var limit = Math.pow(2, 8 * byteLength2 - 1);
    checkInt(this, value, offset, byteLength2, limit - 1, -limit);
  }
  var i = byteLength2 - 1;
  var mul = 1;
  var sub = 0;
  this[offset + i] = value & 255;
  while (--i >= 0 && (mul *= 256)) {
    if (value < 0 && sub === 0 && this[offset + i + 1] !== 0) {
      sub = 1;
    }
    this[offset + i] = (value / mul >> 0) - sub & 255;
  }
  return offset + byteLength2;
};
Buffer.prototype.writeInt8 = function writeInt8(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 1, 127, -128);
  if (!Buffer.TYPED_ARRAY_SUPPORT)
    value = Math.floor(value);
  if (value < 0)
    value = 255 + value + 1;
  this[offset] = value & 255;
  return offset + 1;
};
Buffer.prototype.writeInt16LE = function writeInt16LE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 2, 32767, -32768);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
  } else {
    objectWriteUInt16(this, value, offset, true);
  }
  return offset + 2;
};
Buffer.prototype.writeInt16BE = function writeInt16BE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 2, 32767, -32768);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value >>> 8;
    this[offset + 1] = value & 255;
  } else {
    objectWriteUInt16(this, value, offset, false);
  }
  return offset + 2;
};
Buffer.prototype.writeInt32LE = function writeInt32LE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 4, 2147483647, -2147483648);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value & 255;
    this[offset + 1] = value >>> 8;
    this[offset + 2] = value >>> 16;
    this[offset + 3] = value >>> 24;
  } else {
    objectWriteUInt32(this, value, offset, true);
  }
  return offset + 4;
};
Buffer.prototype.writeInt32BE = function writeInt32BE(value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert)
    checkInt(this, value, offset, 4, 2147483647, -2147483648);
  if (value < 0)
    value = 4294967295 + value + 1;
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = value >>> 24;
    this[offset + 1] = value >>> 16;
    this[offset + 2] = value >>> 8;
    this[offset + 3] = value & 255;
  } else {
    objectWriteUInt32(this, value, offset, false);
  }
  return offset + 4;
};
function checkIEEE754(buf, value, offset, ext, max, min) {
  if (offset + ext > buf.length)
    throw new RangeError("Index out of range");
  if (offset < 0)
    throw new RangeError("Index out of range");
}
function writeFloat(buf, value, offset, littleEndian, noAssert) {
  if (!noAssert) {
    checkIEEE754(buf, value, offset, 4);
  }
  write(buf, value, offset, littleEndian, 23, 4);
  return offset + 4;
}
Buffer.prototype.writeFloatLE = function writeFloatLE(value, offset, noAssert) {
  return writeFloat(this, value, offset, true, noAssert);
};
Buffer.prototype.writeFloatBE = function writeFloatBE(value, offset, noAssert) {
  return writeFloat(this, value, offset, false, noAssert);
};
function writeDouble(buf, value, offset, littleEndian, noAssert) {
  if (!noAssert) {
    checkIEEE754(buf, value, offset, 8);
  }
  write(buf, value, offset, littleEndian, 52, 8);
  return offset + 8;
}
Buffer.prototype.writeDoubleLE = function writeDoubleLE(value, offset, noAssert) {
  return writeDouble(this, value, offset, true, noAssert);
};
Buffer.prototype.writeDoubleBE = function writeDoubleBE(value, offset, noAssert) {
  return writeDouble(this, value, offset, false, noAssert);
};
Buffer.prototype.copy = function copy(target, targetStart, start, end) {
  if (!start)
    start = 0;
  if (!end && end !== 0)
    end = this.length;
  if (targetStart >= target.length)
    targetStart = target.length;
  if (!targetStart)
    targetStart = 0;
  if (end > 0 && end < start)
    end = start;
  if (end === start)
    return 0;
  if (target.length === 0 || this.length === 0)
    return 0;
  if (targetStart < 0) {
    throw new RangeError("targetStart out of bounds");
  }
  if (start < 0 || start >= this.length)
    throw new RangeError("sourceStart out of bounds");
  if (end < 0)
    throw new RangeError("sourceEnd out of bounds");
  if (end > this.length)
    end = this.length;
  if (target.length - targetStart < end - start) {
    end = target.length - targetStart + start;
  }
  var len = end - start;
  var i;
  if (this === target && start < targetStart && targetStart < end) {
    for (i = len - 1; i >= 0; --i) {
      target[i + targetStart] = this[i + start];
    }
  } else if (len < 1e3 || !Buffer.TYPED_ARRAY_SUPPORT) {
    for (i = 0; i < len; ++i) {
      target[i + targetStart] = this[i + start];
    }
  } else {
    Uint8Array.prototype.set.call(
      target,
      this.subarray(start, start + len),
      targetStart
    );
  }
  return len;
};
Buffer.prototype.fill = function fill(val, start, end, encoding) {
  if (typeof val === "string") {
    if (typeof start === "string") {
      encoding = start;
      start = 0;
      end = this.length;
    } else if (typeof end === "string") {
      encoding = end;
      end = this.length;
    }
    if (val.length === 1) {
      var code = val.charCodeAt(0);
      if (code < 256) {
        val = code;
      }
    }
    if (encoding !== void 0 && typeof encoding !== "string") {
      throw new TypeError("encoding must be a string");
    }
    if (typeof encoding === "string" && !Buffer.isEncoding(encoding)) {
      throw new TypeError("Unknown encoding: " + encoding);
    }
  } else if (typeof val === "number") {
    val = val & 255;
  }
  if (start < 0 || this.length < start || this.length < end) {
    throw new RangeError("Out of range index");
  }
  if (end <= start) {
    return this;
  }
  start = start >>> 0;
  end = end === void 0 ? this.length : end >>> 0;
  if (!val)
    val = 0;
  var i;
  if (typeof val === "number") {
    for (i = start; i < end; ++i) {
      this[i] = val;
    }
  } else {
    var bytes = internalIsBuffer(val) ? val : utf8ToBytes(new Buffer(val, encoding).toString());
    var len = bytes.length;
    for (i = 0; i < end - start; ++i) {
      this[i + start] = bytes[i % len];
    }
  }
  return this;
};
var INVALID_BASE64_RE = /[^+\/0-9A-Za-z-_]/g;
function base64clean(str) {
  str = stringtrim(str).replace(INVALID_BASE64_RE, "");
  if (str.length < 2)
    return "";
  while (str.length % 4 !== 0) {
    str = str + "=";
  }
  return str;
}
function stringtrim(str) {
  if (str.trim)
    return str.trim();
  return str.replace(/^\s+|\s+$/g, "");
}
function toHex(n) {
  if (n < 16)
    return "0" + n.toString(16);
  return n.toString(16);
}
function utf8ToBytes(string, units) {
  units = units || Infinity;
  var codePoint;
  var length = string.length;
  var leadSurrogate = null;
  var bytes = [];
  for (var i = 0; i < length; ++i) {
    codePoint = string.charCodeAt(i);
    if (codePoint > 55295 && codePoint < 57344) {
      if (!leadSurrogate) {
        if (codePoint > 56319) {
          if ((units -= 3) > -1)
            bytes.push(239, 191, 189);
          continue;
        } else if (i + 1 === length) {
          if ((units -= 3) > -1)
            bytes.push(239, 191, 189);
          continue;
        }
        leadSurrogate = codePoint;
        continue;
      }
      if (codePoint < 56320) {
        if ((units -= 3) > -1)
          bytes.push(239, 191, 189);
        leadSurrogate = codePoint;
        continue;
      }
      codePoint = (leadSurrogate - 55296 << 10 | codePoint - 56320) + 65536;
    } else if (leadSurrogate) {
      if ((units -= 3) > -1)
        bytes.push(239, 191, 189);
    }
    leadSurrogate = null;
    if (codePoint < 128) {
      if ((units -= 1) < 0)
        break;
      bytes.push(codePoint);
    } else if (codePoint < 2048) {
      if ((units -= 2) < 0)
        break;
      bytes.push(
        codePoint >> 6 | 192,
        codePoint & 63 | 128
      );
    } else if (codePoint < 65536) {
      if ((units -= 3) < 0)
        break;
      bytes.push(
        codePoint >> 12 | 224,
        codePoint >> 6 & 63 | 128,
        codePoint & 63 | 128
      );
    } else if (codePoint < 1114112) {
      if ((units -= 4) < 0)
        break;
      bytes.push(
        codePoint >> 18 | 240,
        codePoint >> 12 & 63 | 128,
        codePoint >> 6 & 63 | 128,
        codePoint & 63 | 128
      );
    } else {
      throw new Error("Invalid code point");
    }
  }
  return bytes;
}
function asciiToBytes(str) {
  var byteArray = [];
  for (var i = 0; i < str.length; ++i) {
    byteArray.push(str.charCodeAt(i) & 255);
  }
  return byteArray;
}
function utf16leToBytes(str, units) {
  var c, hi, lo;
  var byteArray = [];
  for (var i = 0; i < str.length; ++i) {
    if ((units -= 2) < 0)
      break;
    c = str.charCodeAt(i);
    hi = c >> 8;
    lo = c % 256;
    byteArray.push(lo);
    byteArray.push(hi);
  }
  return byteArray;
}
function base64ToBytes(str) {
  return toByteArray(base64clean(str));
}
function blitBuffer(src, dst, offset, length) {
  for (var i = 0; i < length; ++i) {
    if (i + offset >= dst.length || i >= src.length)
      break;
    dst[i + offset] = src[i];
  }
  return i;
}
function isnan(val) {
  return val !== val;
}
function isBuffer(obj) {
  return obj != null && (!!obj._isBuffer || isFastBuffer(obj) || isSlowBuffer(obj));
}
function isFastBuffer(obj) {
  return !!obj.constructor && typeof obj.constructor.isBuffer === "function" && obj.constructor.isBuffer(obj);
}
function isSlowBuffer(obj) {
  return typeof obj.readFloatLE === "function" && typeof obj.slice === "function" && isFastBuffer(obj.slice(0, 0));
}
var CsvError = class extends Error {
  constructor(code, message, options, ...contexts) {
    if (Array.isArray(message))
      message = message.join(" ").trim();
    super(message);
    if (Error.captureStackTrace !== void 0) {
      Error.captureStackTrace(this, CsvError);
    }
    this.code = code;
    for (const context of contexts) {
      for (const key in context) {
        const value = context[key];
        this[key] = isBuffer(value) ? value.toString(options.encoding) : value == null ? value : JSON.parse(JSON.stringify(value));
      }
    }
  }
};
var is_object = function(obj) {
  return typeof obj === "object" && obj !== null && !Array.isArray(obj);
};
var normalize_columns_array = function(columns) {
  const normalizedColumns = [];
  for (let i = 0, l = columns.length; i < l; i++) {
    const column = columns[i];
    if (column === void 0 || column === null || column === false) {
      normalizedColumns[i] = { disabled: true };
    } else if (typeof column === "string") {
      normalizedColumns[i] = { name: column };
    } else if (is_object(column)) {
      if (typeof column.name !== "string") {
        throw new CsvError("CSV_OPTION_COLUMNS_MISSING_NAME", [
          "Option columns missing name:",
          `property "name" is required at position ${i}`,
          "when column is an object literal"
        ]);
      }
      normalizedColumns[i] = column;
    } else {
      throw new CsvError("CSV_INVALID_COLUMN_DEFINITION", [
        "Invalid column definition:",
        "expect a string or a literal object,",
        `got ${JSON.stringify(column)} at position ${i}`
      ]);
    }
  }
  return normalizedColumns;
};
var ResizeableBuffer = class {
  constructor(size = 100) {
    this.size = size;
    this.length = 0;
    this.buf = Buffer.allocUnsafe(size);
  }
  prepend(val) {
    if (isBuffer(val)) {
      const length = this.length + val.length;
      if (length >= this.size) {
        this.resize();
        if (length >= this.size) {
          throw Error("INVALID_BUFFER_STATE");
        }
      }
      const buf = this.buf;
      this.buf = Buffer.allocUnsafe(this.size);
      val.copy(this.buf, 0);
      buf.copy(this.buf, val.length);
      this.length += val.length;
    } else {
      const length = this.length++;
      if (length === this.size) {
        this.resize();
      }
      const buf = this.clone();
      this.buf[0] = val;
      buf.copy(this.buf, 1, 0, length);
    }
  }
  append(val) {
    const length = this.length++;
    if (length === this.size) {
      this.resize();
    }
    this.buf[length] = val;
  }
  clone() {
    return Buffer.from(this.buf.slice(0, this.length));
  }
  resize() {
    const length = this.length;
    this.size = this.size * 2;
    const buf = Buffer.allocUnsafe(this.size);
    this.buf.copy(buf, 0, 0, length);
    this.buf = buf;
  }
  toString(encoding) {
    if (encoding) {
      return this.buf.slice(0, this.length).toString(encoding);
    } else {
      return Uint8Array.prototype.slice.call(this.buf.slice(0, this.length));
    }
  }
  toJSON() {
    return this.toString("utf8");
  }
  reset() {
    this.length = 0;
  }
};
var np = 12;
var cr$1 = 13;
var nl$1 = 10;
var space = 32;
var tab = 9;
var init_state = function(options) {
  return {
    bomSkipped: false,
    bufBytesStart: 0,
    castField: options.cast_function,
    commenting: false,
    // Current error encountered by a record
    error: void 0,
    enabled: options.from_line === 1,
    escaping: false,
    escapeIsQuote: isBuffer(options.escape) && isBuffer(options.quote) && Buffer.compare(options.escape, options.quote) === 0,
    // columns can be `false`, `true`, `Array`
    expectedRecordLength: Array.isArray(options.columns) ? options.columns.length : void 0,
    field: new ResizeableBuffer(20),
    firstLineToHeaders: options.cast_first_line_to_header,
    needMoreDataSize: Math.max(
      // Skip if the remaining buffer smaller than comment
      options.comment !== null ? options.comment.length : 0,
      ...options.delimiter.map((delimiter) => delimiter.length),
      // Skip if the remaining buffer can be escape sequence
      options.quote !== null ? options.quote.length : 0
    ),
    previousBuf: void 0,
    quoting: false,
    stop: false,
    rawBuffer: new ResizeableBuffer(100),
    record: [],
    recordHasError: false,
    record_length: 0,
    recordDelimiterMaxLength: options.record_delimiter.length === 0 ? 0 : Math.max(...options.record_delimiter.map((v) => v.length)),
    trimChars: [
      Buffer.from(" ", options.encoding)[0],
      Buffer.from("	", options.encoding)[0]
    ],
    wasQuoting: false,
    wasRowDelimiter: false,
    timchars: [
      Buffer.from(Buffer.from([cr$1], "utf8").toString(), options.encoding),
      Buffer.from(Buffer.from([nl$1], "utf8").toString(), options.encoding),
      Buffer.from(Buffer.from([np], "utf8").toString(), options.encoding),
      Buffer.from(Buffer.from([space], "utf8").toString(), options.encoding),
      Buffer.from(Buffer.from([tab], "utf8").toString(), options.encoding)
    ]
  };
};
var underscore = function(str) {
  return str.replace(/([A-Z])/g, function(_, match) {
    return "_" + match.toLowerCase();
  });
};
var normalize_options = function(opts) {
  const options = {};
  for (const opt in opts) {
    options[underscore(opt)] = opts[opt];
  }
  if (options.encoding === void 0 || options.encoding === true) {
    options.encoding = "utf8";
  } else if (options.encoding === null || options.encoding === false) {
    options.encoding = null;
  } else if (typeof options.encoding !== "string" && options.encoding !== null) {
    throw new CsvError(
      "CSV_INVALID_OPTION_ENCODING",
      [
        "Invalid option encoding:",
        "encoding must be a string or null to return a buffer,",
        `got ${JSON.stringify(options.encoding)}`
      ],
      options
    );
  }
  if (options.bom === void 0 || options.bom === null || options.bom === false) {
    options.bom = false;
  } else if (options.bom !== true) {
    throw new CsvError(
      "CSV_INVALID_OPTION_BOM",
      [
        "Invalid option bom:",
        "bom must be true,",
        `got ${JSON.stringify(options.bom)}`
      ],
      options
    );
  }
  options.cast_function = null;
  if (options.cast === void 0 || options.cast === null || options.cast === false || options.cast === "") {
    options.cast = void 0;
  } else if (typeof options.cast === "function") {
    options.cast_function = options.cast;
    options.cast = true;
  } else if (options.cast !== true) {
    throw new CsvError(
      "CSV_INVALID_OPTION_CAST",
      [
        "Invalid option cast:",
        "cast must be true or a function,",
        `got ${JSON.stringify(options.cast)}`
      ],
      options
    );
  }
  if (options.cast_date === void 0 || options.cast_date === null || options.cast_date === false || options.cast_date === "") {
    options.cast_date = false;
  } else if (options.cast_date === true) {
    options.cast_date = function(value) {
      const date = Date.parse(value);
      return !isNaN(date) ? new Date(date) : value;
    };
  } else if (typeof options.cast_date !== "function") {
    throw new CsvError(
      "CSV_INVALID_OPTION_CAST_DATE",
      [
        "Invalid option cast_date:",
        "cast_date must be true or a function,",
        `got ${JSON.stringify(options.cast_date)}`
      ],
      options
    );
  }
  options.cast_first_line_to_header = null;
  if (options.columns === true) {
    options.cast_first_line_to_header = void 0;
  } else if (typeof options.columns === "function") {
    options.cast_first_line_to_header = options.columns;
    options.columns = true;
  } else if (Array.isArray(options.columns)) {
    options.columns = normalize_columns_array(options.columns);
  } else if (options.columns === void 0 || options.columns === null || options.columns === false) {
    options.columns = false;
  } else {
    throw new CsvError(
      "CSV_INVALID_OPTION_COLUMNS",
      [
        "Invalid option columns:",
        "expect an array, a function or true,",
        `got ${JSON.stringify(options.columns)}`
      ],
      options
    );
  }
  if (options.group_columns_by_name === void 0 || options.group_columns_by_name === null || options.group_columns_by_name === false) {
    options.group_columns_by_name = false;
  } else if (options.group_columns_by_name !== true) {
    throw new CsvError(
      "CSV_INVALID_OPTION_GROUP_COLUMNS_BY_NAME",
      [
        "Invalid option group_columns_by_name:",
        "expect an boolean,",
        `got ${JSON.stringify(options.group_columns_by_name)}`
      ],
      options
    );
  } else if (options.columns === false) {
    throw new CsvError(
      "CSV_INVALID_OPTION_GROUP_COLUMNS_BY_NAME",
      [
        "Invalid option group_columns_by_name:",
        "the `columns` mode must be activated."
      ],
      options
    );
  }
  if (options.comment === void 0 || options.comment === null || options.comment === false || options.comment === "") {
    options.comment = null;
  } else {
    if (typeof options.comment === "string") {
      options.comment = Buffer.from(options.comment, options.encoding);
    }
    if (!isBuffer(options.comment)) {
      throw new CsvError(
        "CSV_INVALID_OPTION_COMMENT",
        [
          "Invalid option comment:",
          "comment must be a buffer or a string,",
          `got ${JSON.stringify(options.comment)}`
        ],
        options
      );
    }
  }
  if (options.comment_no_infix === void 0 || options.comment_no_infix === null || options.comment_no_infix === false) {
    options.comment_no_infix = false;
  } else if (options.comment_no_infix !== true) {
    throw new CsvError(
      "CSV_INVALID_OPTION_COMMENT",
      [
        "Invalid option comment_no_infix:",
        "value must be a boolean,",
        `got ${JSON.stringify(options.comment_no_infix)}`
      ],
      options
    );
  }
  const delimiter_json = JSON.stringify(options.delimiter);
  if (!Array.isArray(options.delimiter))
    options.delimiter = [options.delimiter];
  if (options.delimiter.length === 0) {
    throw new CsvError(
      "CSV_INVALID_OPTION_DELIMITER",
      [
        "Invalid option delimiter:",
        "delimiter must be a non empty string or buffer or array of string|buffer,",
        `got ${delimiter_json}`
      ],
      options
    );
  }
  options.delimiter = options.delimiter.map(function(delimiter) {
    if (delimiter === void 0 || delimiter === null || delimiter === false) {
      return Buffer.from(",", options.encoding);
    }
    if (typeof delimiter === "string") {
      delimiter = Buffer.from(delimiter, options.encoding);
    }
    if (!isBuffer(delimiter) || delimiter.length === 0) {
      throw new CsvError(
        "CSV_INVALID_OPTION_DELIMITER",
        [
          "Invalid option delimiter:",
          "delimiter must be a non empty string or buffer or array of string|buffer,",
          `got ${delimiter_json}`
        ],
        options
      );
    }
    return delimiter;
  });
  if (options.escape === void 0 || options.escape === true) {
    options.escape = Buffer.from('"', options.encoding);
  } else if (typeof options.escape === "string") {
    options.escape = Buffer.from(options.escape, options.encoding);
  } else if (options.escape === null || options.escape === false) {
    options.escape = null;
  }
  if (options.escape !== null) {
    if (!isBuffer(options.escape)) {
      throw new Error(
        `Invalid Option: escape must be a buffer, a string or a boolean, got ${JSON.stringify(options.escape)}`
      );
    }
  }
  if (options.from === void 0 || options.from === null) {
    options.from = 1;
  } else {
    if (typeof options.from === "string" && /\d+/.test(options.from)) {
      options.from = parseInt(options.from);
    }
    if (Number.isInteger(options.from)) {
      if (options.from < 0) {
        throw new Error(
          `Invalid Option: from must be a positive integer, got ${JSON.stringify(opts.from)}`
        );
      }
    } else {
      throw new Error(
        `Invalid Option: from must be an integer, got ${JSON.stringify(options.from)}`
      );
    }
  }
  if (options.from_line === void 0 || options.from_line === null) {
    options.from_line = 1;
  } else {
    if (typeof options.from_line === "string" && /\d+/.test(options.from_line)) {
      options.from_line = parseInt(options.from_line);
    }
    if (Number.isInteger(options.from_line)) {
      if (options.from_line <= 0) {
        throw new Error(
          `Invalid Option: from_line must be a positive integer greater than 0, got ${JSON.stringify(opts.from_line)}`
        );
      }
    } else {
      throw new Error(
        `Invalid Option: from_line must be an integer, got ${JSON.stringify(opts.from_line)}`
      );
    }
  }
  if (options.ignore_last_delimiters === void 0 || options.ignore_last_delimiters === null) {
    options.ignore_last_delimiters = false;
  } else if (typeof options.ignore_last_delimiters === "number") {
    options.ignore_last_delimiters = Math.floor(options.ignore_last_delimiters);
    if (options.ignore_last_delimiters === 0) {
      options.ignore_last_delimiters = false;
    }
  } else if (typeof options.ignore_last_delimiters !== "boolean") {
    throw new CsvError(
      "CSV_INVALID_OPTION_IGNORE_LAST_DELIMITERS",
      [
        "Invalid option `ignore_last_delimiters`:",
        "the value must be a boolean value or an integer,",
        `got ${JSON.stringify(options.ignore_last_delimiters)}`
      ],
      options
    );
  }
  if (options.ignore_last_delimiters === true && options.columns === false) {
    throw new CsvError(
      "CSV_IGNORE_LAST_DELIMITERS_REQUIRES_COLUMNS",
      [
        "The option `ignore_last_delimiters`",
        "requires the activation of the `columns` option"
      ],
      options
    );
  }
  if (options.info === void 0 || options.info === null || options.info === false) {
    options.info = false;
  } else if (options.info !== true) {
    throw new Error(
      `Invalid Option: info must be true, got ${JSON.stringify(options.info)}`
    );
  }
  if (options.max_record_size === void 0 || options.max_record_size === null || options.max_record_size === false) {
    options.max_record_size = 0;
  } else if (Number.isInteger(options.max_record_size) && options.max_record_size >= 0)
    ;
  else if (typeof options.max_record_size === "string" && /\d+/.test(options.max_record_size)) {
    options.max_record_size = parseInt(options.max_record_size);
  } else {
    throw new Error(
      `Invalid Option: max_record_size must be a positive integer, got ${JSON.stringify(options.max_record_size)}`
    );
  }
  if (options.objname === void 0 || options.objname === null || options.objname === false) {
    options.objname = void 0;
  } else if (isBuffer(options.objname)) {
    if (options.objname.length === 0) {
      throw new Error(`Invalid Option: objname must be a non empty buffer`);
    }
    if (options.encoding === null)
      ;
    else {
      options.objname = options.objname.toString(options.encoding);
    }
  } else if (typeof options.objname === "string") {
    if (options.objname.length === 0) {
      throw new Error(`Invalid Option: objname must be a non empty string`);
    }
  } else if (typeof options.objname === "number")
    ;
  else {
    throw new Error(
      `Invalid Option: objname must be a string or a buffer, got ${options.objname}`
    );
  }
  if (options.objname !== void 0) {
    if (typeof options.objname === "number") {
      if (options.columns !== false) {
        throw Error(
          "Invalid Option: objname index cannot be combined with columns or be defined as a field"
        );
      }
    } else {
      if (options.columns === false) {
        throw Error(
          "Invalid Option: objname field must be combined with columns or be defined as an index"
        );
      }
    }
  }
  if (options.on_record === void 0 || options.on_record === null) {
    options.on_record = void 0;
  } else if (typeof options.on_record !== "function") {
    throw new CsvError(
      "CSV_INVALID_OPTION_ON_RECORD",
      [
        "Invalid option `on_record`:",
        "expect a function,",
        `got ${JSON.stringify(options.on_record)}`
      ],
      options
    );
  }
  if (options.on_skip !== void 0 && options.on_skip !== null && typeof options.on_skip !== "function") {
    throw new Error(
      `Invalid Option: on_skip must be a function, got ${JSON.stringify(options.on_skip)}`
    );
  }
  if (options.quote === null || options.quote === false || options.quote === "") {
    options.quote = null;
  } else {
    if (options.quote === void 0 || options.quote === true) {
      options.quote = Buffer.from('"', options.encoding);
    } else if (typeof options.quote === "string") {
      options.quote = Buffer.from(options.quote, options.encoding);
    }
    if (!isBuffer(options.quote)) {
      throw new Error(
        `Invalid Option: quote must be a buffer or a string, got ${JSON.stringify(options.quote)}`
      );
    }
  }
  if (options.raw === void 0 || options.raw === null || options.raw === false) {
    options.raw = false;
  } else if (options.raw !== true) {
    throw new Error(
      `Invalid Option: raw must be true, got ${JSON.stringify(options.raw)}`
    );
  }
  if (options.record_delimiter === void 0) {
    options.record_delimiter = [];
  } else if (typeof options.record_delimiter === "string" || isBuffer(options.record_delimiter)) {
    if (options.record_delimiter.length === 0) {
      throw new CsvError(
        "CSV_INVALID_OPTION_RECORD_DELIMITER",
        [
          "Invalid option `record_delimiter`:",
          "value must be a non empty string or buffer,",
          `got ${JSON.stringify(options.record_delimiter)}`
        ],
        options
      );
    }
    options.record_delimiter = [options.record_delimiter];
  } else if (!Array.isArray(options.record_delimiter)) {
    throw new CsvError(
      "CSV_INVALID_OPTION_RECORD_DELIMITER",
      [
        "Invalid option `record_delimiter`:",
        "value must be a string, a buffer or array of string|buffer,",
        `got ${JSON.stringify(options.record_delimiter)}`
      ],
      options
    );
  }
  options.record_delimiter = options.record_delimiter.map(function(rd, i) {
    if (typeof rd !== "string" && !isBuffer(rd)) {
      throw new CsvError(
        "CSV_INVALID_OPTION_RECORD_DELIMITER",
        [
          "Invalid option `record_delimiter`:",
          "value must be a string, a buffer or array of string|buffer",
          `at index ${i},`,
          `got ${JSON.stringify(rd)}`
        ],
        options
      );
    } else if (rd.length === 0) {
      throw new CsvError(
        "CSV_INVALID_OPTION_RECORD_DELIMITER",
        [
          "Invalid option `record_delimiter`:",
          "value must be a non empty string or buffer",
          `at index ${i},`,
          `got ${JSON.stringify(rd)}`
        ],
        options
      );
    }
    if (typeof rd === "string") {
      rd = Buffer.from(rd, options.encoding);
    }
    return rd;
  });
  if (typeof options.relax_column_count === "boolean")
    ;
  else if (options.relax_column_count === void 0 || options.relax_column_count === null) {
    options.relax_column_count = false;
  } else {
    throw new Error(
      `Invalid Option: relax_column_count must be a boolean, got ${JSON.stringify(options.relax_column_count)}`
    );
  }
  if (typeof options.relax_column_count_less === "boolean")
    ;
  else if (options.relax_column_count_less === void 0 || options.relax_column_count_less === null) {
    options.relax_column_count_less = false;
  } else {
    throw new Error(
      `Invalid Option: relax_column_count_less must be a boolean, got ${JSON.stringify(options.relax_column_count_less)}`
    );
  }
  if (typeof options.relax_column_count_more === "boolean")
    ;
  else if (options.relax_column_count_more === void 0 || options.relax_column_count_more === null) {
    options.relax_column_count_more = false;
  } else {
    throw new Error(
      `Invalid Option: relax_column_count_more must be a boolean, got ${JSON.stringify(options.relax_column_count_more)}`
    );
  }
  if (typeof options.relax_quotes === "boolean")
    ;
  else if (options.relax_quotes === void 0 || options.relax_quotes === null) {
    options.relax_quotes = false;
  } else {
    throw new Error(
      `Invalid Option: relax_quotes must be a boolean, got ${JSON.stringify(options.relax_quotes)}`
    );
  }
  if (typeof options.skip_empty_lines === "boolean")
    ;
  else if (options.skip_empty_lines === void 0 || options.skip_empty_lines === null) {
    options.skip_empty_lines = false;
  } else {
    throw new Error(
      `Invalid Option: skip_empty_lines must be a boolean, got ${JSON.stringify(options.skip_empty_lines)}`
    );
  }
  if (typeof options.skip_records_with_empty_values === "boolean")
    ;
  else if (options.skip_records_with_empty_values === void 0 || options.skip_records_with_empty_values === null) {
    options.skip_records_with_empty_values = false;
  } else {
    throw new Error(
      `Invalid Option: skip_records_with_empty_values must be a boolean, got ${JSON.stringify(options.skip_records_with_empty_values)}`
    );
  }
  if (typeof options.skip_records_with_error === "boolean")
    ;
  else if (options.skip_records_with_error === void 0 || options.skip_records_with_error === null) {
    options.skip_records_with_error = false;
  } else {
    throw new Error(
      `Invalid Option: skip_records_with_error must be a boolean, got ${JSON.stringify(options.skip_records_with_error)}`
    );
  }
  if (options.rtrim === void 0 || options.rtrim === null || options.rtrim === false) {
    options.rtrim = false;
  } else if (options.rtrim !== true) {
    throw new Error(
      `Invalid Option: rtrim must be a boolean, got ${JSON.stringify(options.rtrim)}`
    );
  }
  if (options.ltrim === void 0 || options.ltrim === null || options.ltrim === false) {
    options.ltrim = false;
  } else if (options.ltrim !== true) {
    throw new Error(
      `Invalid Option: ltrim must be a boolean, got ${JSON.stringify(options.ltrim)}`
    );
  }
  if (options.trim === void 0 || options.trim === null || options.trim === false) {
    options.trim = false;
  } else if (options.trim !== true) {
    throw new Error(
      `Invalid Option: trim must be a boolean, got ${JSON.stringify(options.trim)}`
    );
  }
  if (options.trim === true && opts.ltrim !== false) {
    options.ltrim = true;
  } else if (options.ltrim !== true) {
    options.ltrim = false;
  }
  if (options.trim === true && opts.rtrim !== false) {
    options.rtrim = true;
  } else if (options.rtrim !== true) {
    options.rtrim = false;
  }
  if (options.to === void 0 || options.to === null) {
    options.to = -1;
  } else {
    if (typeof options.to === "string" && /\d+/.test(options.to)) {
      options.to = parseInt(options.to);
    }
    if (Number.isInteger(options.to)) {
      if (options.to <= 0) {
        throw new Error(
          `Invalid Option: to must be a positive integer greater than 0, got ${JSON.stringify(opts.to)}`
        );
      }
    } else {
      throw new Error(
        `Invalid Option: to must be an integer, got ${JSON.stringify(opts.to)}`
      );
    }
  }
  if (options.to_line === void 0 || options.to_line === null) {
    options.to_line = -1;
  } else {
    if (typeof options.to_line === "string" && /\d+/.test(options.to_line)) {
      options.to_line = parseInt(options.to_line);
    }
    if (Number.isInteger(options.to_line)) {
      if (options.to_line <= 0) {
        throw new Error(
          `Invalid Option: to_line must be a positive integer greater than 0, got ${JSON.stringify(opts.to_line)}`
        );
      }
    } else {
      throw new Error(
        `Invalid Option: to_line must be an integer, got ${JSON.stringify(opts.to_line)}`
      );
    }
  }
  return options;
};
var isRecordEmpty = function(record) {
  return record.every(
    (field) => field == null || field.toString && field.toString().trim() === ""
  );
};
var cr = 13;
var nl = 10;
var boms = {
  // Note, the following are equals:
  // Buffer.from("\ufeff")
  // Buffer.from([239, 187, 191])
  // Buffer.from('EFBBBF', 'hex')
  utf8: Buffer.from([239, 187, 191]),
  // Note, the following are equals:
  // Buffer.from "\ufeff", 'utf16le
  // Buffer.from([255, 254])
  utf16le: Buffer.from([255, 254])
};
var transform = function(original_options = {}) {
  const info = {
    bytes: 0,
    comment_lines: 0,
    empty_lines: 0,
    invalid_field_length: 0,
    lines: 1,
    records: 0
  };
  const options = normalize_options(original_options);
  return {
    info,
    original_options,
    options,
    state: init_state(options),
    __needMoreData: function(i, bufLen, end) {
      if (end)
        return false;
      const { encoding, escape, quote } = this.options;
      const { quoting, needMoreDataSize, recordDelimiterMaxLength } = this.state;
      const numOfCharLeft = bufLen - i - 1;
      const requiredLength = Math.max(
        needMoreDataSize,
        // Skip if the remaining buffer smaller than record delimiter
        // If "record_delimiter" is yet to be discovered:
        // 1. It is equals to `[]` and "recordDelimiterMaxLength" equals `0`
        // 2. We set the length to windows line ending in the current encoding
        // Note, that encoding is known from user or bom discovery at that point
        // recordDelimiterMaxLength,
        recordDelimiterMaxLength === 0 ? Buffer.from("\r\n", encoding).length : recordDelimiterMaxLength,
        // Skip if remaining buffer can be an escaped quote
        quoting ? (escape === null ? 0 : escape.length) + quote.length : 0,
        // Skip if remaining buffer can be record delimiter following the closing quote
        quoting ? quote.length + recordDelimiterMaxLength : 0
      );
      return numOfCharLeft < requiredLength;
    },
    // Central parser implementation
    parse: function(nextBuf, end, push, close) {
      const {
        bom,
        comment_no_infix,
        encoding,
        from_line,
        ltrim,
        max_record_size,
        raw,
        relax_quotes,
        rtrim,
        skip_empty_lines,
        to,
        to_line
      } = this.options;
      let { comment, escape, quote, record_delimiter } = this.options;
      const { bomSkipped, previousBuf, rawBuffer, escapeIsQuote } = this.state;
      let buf;
      if (previousBuf === void 0) {
        if (nextBuf === void 0) {
          close();
          return;
        } else {
          buf = nextBuf;
        }
      } else if (previousBuf !== void 0 && nextBuf === void 0) {
        buf = previousBuf;
      } else {
        buf = Buffer.concat([previousBuf, nextBuf]);
      }
      if (bomSkipped === false) {
        if (bom === false) {
          this.state.bomSkipped = true;
        } else if (buf.length < 3) {
          if (end === false) {
            this.state.previousBuf = buf;
            return;
          }
        } else {
          for (const encoding2 in boms) {
            if (boms[encoding2].compare(buf, 0, boms[encoding2].length) === 0) {
              const bomLength = boms[encoding2].length;
              this.state.bufBytesStart += bomLength;
              buf = buf.slice(bomLength);
              this.options = normalize_options({
                ...this.original_options,
                encoding: encoding2
              });
              ({ comment, escape, quote } = this.options);
              break;
            }
          }
          this.state.bomSkipped = true;
        }
      }
      const bufLen = buf.length;
      let pos;
      for (pos = 0; pos < bufLen; pos++) {
        if (this.__needMoreData(pos, bufLen, end)) {
          break;
        }
        if (this.state.wasRowDelimiter === true) {
          this.info.lines++;
          this.state.wasRowDelimiter = false;
        }
        if (to_line !== -1 && this.info.lines > to_line) {
          this.state.stop = true;
          close();
          return;
        }
        if (this.state.quoting === false && record_delimiter.length === 0) {
          const record_delimiterCount = this.__autoDiscoverRecordDelimiter(
            buf,
            pos
          );
          if (record_delimiterCount) {
            record_delimiter = this.options.record_delimiter;
          }
        }
        const chr = buf[pos];
        if (raw === true) {
          rawBuffer.append(chr);
        }
        if ((chr === cr || chr === nl) && this.state.wasRowDelimiter === false) {
          this.state.wasRowDelimiter = true;
        }
        if (this.state.escaping === true) {
          this.state.escaping = false;
        } else {
          if (escape !== null && this.state.quoting === true && this.__isEscape(buf, pos, chr) && pos + escape.length < bufLen) {
            if (escapeIsQuote) {
              if (this.__isQuote(buf, pos + escape.length)) {
                this.state.escaping = true;
                pos += escape.length - 1;
                continue;
              }
            } else {
              this.state.escaping = true;
              pos += escape.length - 1;
              continue;
            }
          }
          if (this.state.commenting === false && this.__isQuote(buf, pos)) {
            if (this.state.quoting === true) {
              const nextChr = buf[pos + quote.length];
              const isNextChrTrimable = rtrim && this.__isCharTrimable(buf, pos + quote.length);
              const isNextChrComment = comment !== null && this.__compareBytes(comment, buf, pos + quote.length, nextChr);
              const isNextChrDelimiter = this.__isDelimiter(
                buf,
                pos + quote.length,
                nextChr
              );
              const isNextChrRecordDelimiter = record_delimiter.length === 0 ? this.__autoDiscoverRecordDelimiter(buf, pos + quote.length) : this.__isRecordDelimiter(nextChr, buf, pos + quote.length);
              if (escape !== null && this.__isEscape(buf, pos, chr) && this.__isQuote(buf, pos + escape.length)) {
                pos += escape.length - 1;
              } else if (!nextChr || isNextChrDelimiter || isNextChrRecordDelimiter || isNextChrComment || isNextChrTrimable) {
                this.state.quoting = false;
                this.state.wasQuoting = true;
                pos += quote.length - 1;
                continue;
              } else if (relax_quotes === false) {
                const err = this.__error(
                  new CsvError(
                    "CSV_INVALID_CLOSING_QUOTE",
                    [
                      "Invalid Closing Quote:",
                      `got "${String.fromCharCode(nextChr)}"`,
                      `at line ${this.info.lines}`,
                      "instead of delimiter, record delimiter, trimable character",
                      "(if activated) or comment"
                    ],
                    this.options,
                    this.__infoField()
                  )
                );
                if (err !== void 0)
                  return err;
              } else {
                this.state.quoting = false;
                this.state.wasQuoting = true;
                this.state.field.prepend(quote);
                pos += quote.length - 1;
              }
            } else {
              if (this.state.field.length !== 0) {
                if (relax_quotes === false) {
                  const info2 = this.__infoField();
                  const bom2 = Object.keys(boms).map(
                    (b) => boms[b].equals(this.state.field.toString()) ? b : false
                  ).filter(Boolean)[0];
                  const err = this.__error(
                    new CsvError(
                      "INVALID_OPENING_QUOTE",
                      [
                        "Invalid Opening Quote:",
                        `a quote is found on field ${JSON.stringify(info2.column)} at line ${info2.lines}, value is ${JSON.stringify(this.state.field.toString(encoding))}`,
                        bom2 ? `(${bom2} bom)` : void 0
                      ],
                      this.options,
                      info2,
                      {
                        field: this.state.field
                      }
                    )
                  );
                  if (err !== void 0)
                    return err;
                }
              } else {
                this.state.quoting = true;
                pos += quote.length - 1;
                continue;
              }
            }
          }
          if (this.state.quoting === false) {
            const recordDelimiterLength = this.__isRecordDelimiter(
              chr,
              buf,
              pos
            );
            if (recordDelimiterLength !== 0) {
              const skipCommentLine = this.state.commenting && this.state.wasQuoting === false && this.state.record.length === 0 && this.state.field.length === 0;
              if (skipCommentLine) {
                this.info.comment_lines++;
              } else {
                if (this.state.enabled === false && this.info.lines + (this.state.wasRowDelimiter === true ? 1 : 0) >= from_line) {
                  this.state.enabled = true;
                  this.__resetField();
                  this.__resetRecord();
                  pos += recordDelimiterLength - 1;
                  continue;
                }
                if (skip_empty_lines === true && this.state.wasQuoting === false && this.state.record.length === 0 && this.state.field.length === 0) {
                  this.info.empty_lines++;
                  pos += recordDelimiterLength - 1;
                  continue;
                }
                this.info.bytes = this.state.bufBytesStart + pos;
                const errField = this.__onField();
                if (errField !== void 0)
                  return errField;
                this.info.bytes = this.state.bufBytesStart + pos + recordDelimiterLength;
                const errRecord = this.__onRecord(push);
                if (errRecord !== void 0)
                  return errRecord;
                if (to !== -1 && this.info.records >= to) {
                  this.state.stop = true;
                  close();
                  return;
                }
              }
              this.state.commenting = false;
              pos += recordDelimiterLength - 1;
              continue;
            }
            if (this.state.commenting) {
              continue;
            }
            if (comment !== null && (comment_no_infix === false || this.state.record.length === 0 && this.state.field.length === 0)) {
              const commentCount = this.__compareBytes(comment, buf, pos, chr);
              if (commentCount !== 0) {
                this.state.commenting = true;
                continue;
              }
            }
            const delimiterLength = this.__isDelimiter(buf, pos, chr);
            if (delimiterLength !== 0) {
              this.info.bytes = this.state.bufBytesStart + pos;
              const errField = this.__onField();
              if (errField !== void 0)
                return errField;
              pos += delimiterLength - 1;
              continue;
            }
          }
        }
        if (this.state.commenting === false) {
          if (max_record_size !== 0 && this.state.record_length + this.state.field.length > max_record_size) {
            return this.__error(
              new CsvError(
                "CSV_MAX_RECORD_SIZE",
                [
                  "Max Record Size:",
                  "record exceed the maximum number of tolerated bytes",
                  `of ${max_record_size}`,
                  `at line ${this.info.lines}`
                ],
                this.options,
                this.__infoField()
              )
            );
          }
        }
        const lappend = ltrim === false || this.state.quoting === true || this.state.field.length !== 0 || !this.__isCharTrimable(buf, pos);
        const rappend = rtrim === false || this.state.wasQuoting === false;
        if (lappend === true && rappend === true) {
          this.state.field.append(chr);
        } else if (rtrim === true && !this.__isCharTrimable(buf, pos)) {
          return this.__error(
            new CsvError(
              "CSV_NON_TRIMABLE_CHAR_AFTER_CLOSING_QUOTE",
              [
                "Invalid Closing Quote:",
                "found non trimable byte after quote",
                `at line ${this.info.lines}`
              ],
              this.options,
              this.__infoField()
            )
          );
        } else {
          if (lappend === false) {
            pos += this.__isCharTrimable(buf, pos) - 1;
          }
          continue;
        }
      }
      if (end === true) {
        if (this.state.quoting === true) {
          const err = this.__error(
            new CsvError(
              "CSV_QUOTE_NOT_CLOSED",
              [
                "Quote Not Closed:",
                `the parsing is finished with an opening quote at line ${this.info.lines}`
              ],
              this.options,
              this.__infoField()
            )
          );
          if (err !== void 0)
            return err;
        } else {
          if (this.state.wasQuoting === true || this.state.record.length !== 0 || this.state.field.length !== 0) {
            this.info.bytes = this.state.bufBytesStart + pos;
            const errField = this.__onField();
            if (errField !== void 0)
              return errField;
            const errRecord = this.__onRecord(push);
            if (errRecord !== void 0)
              return errRecord;
          } else if (this.state.wasRowDelimiter === true) {
            this.info.empty_lines++;
          } else if (this.state.commenting === true) {
            this.info.comment_lines++;
          }
        }
      } else {
        this.state.bufBytesStart += pos;
        this.state.previousBuf = buf.slice(pos);
      }
      if (this.state.wasRowDelimiter === true) {
        this.info.lines++;
        this.state.wasRowDelimiter = false;
      }
    },
    __onRecord: function(push) {
      const {
        columns,
        group_columns_by_name,
        encoding,
        info: info2,
        from: from2,
        relax_column_count,
        relax_column_count_less,
        relax_column_count_more,
        raw,
        skip_records_with_empty_values
      } = this.options;
      const { enabled, record } = this.state;
      if (enabled === false) {
        return this.__resetRecord();
      }
      const recordLength = record.length;
      if (columns === true) {
        if (skip_records_with_empty_values === true && isRecordEmpty(record)) {
          this.__resetRecord();
          return;
        }
        return this.__firstLineToColumns(record);
      }
      if (columns === false && this.info.records === 0) {
        this.state.expectedRecordLength = recordLength;
      }
      if (recordLength !== this.state.expectedRecordLength) {
        const err = columns === false ? new CsvError(
          "CSV_RECORD_INCONSISTENT_FIELDS_LENGTH",
          [
            "Invalid Record Length:",
            `expect ${this.state.expectedRecordLength},`,
            `got ${recordLength} on line ${this.info.lines}`
          ],
          this.options,
          this.__infoField(),
          {
            record
          }
        ) : new CsvError(
          "CSV_RECORD_INCONSISTENT_COLUMNS",
          [
            "Invalid Record Length:",
            `columns length is ${columns.length},`,
            // rename columns
            `got ${recordLength} on line ${this.info.lines}`
          ],
          this.options,
          this.__infoField(),
          {
            record
          }
        );
        if (relax_column_count === true || relax_column_count_less === true && recordLength < this.state.expectedRecordLength || relax_column_count_more === true && recordLength > this.state.expectedRecordLength) {
          this.info.invalid_field_length++;
          this.state.error = err;
        } else {
          const finalErr = this.__error(err);
          if (finalErr)
            return finalErr;
        }
      }
      if (skip_records_with_empty_values === true && isRecordEmpty(record)) {
        this.__resetRecord();
        return;
      }
      if (this.state.recordHasError === true) {
        this.__resetRecord();
        this.state.recordHasError = false;
        return;
      }
      this.info.records++;
      if (from2 === 1 || this.info.records >= from2) {
        const { objname } = this.options;
        if (columns !== false) {
          const obj = {};
          for (let i = 0, l = record.length; i < l; i++) {
            if (columns[i] === void 0 || columns[i].disabled)
              continue;
            if (group_columns_by_name === true && obj[columns[i].name] !== void 0) {
              if (Array.isArray(obj[columns[i].name])) {
                obj[columns[i].name] = obj[columns[i].name].concat(record[i]);
              } else {
                obj[columns[i].name] = [obj[columns[i].name], record[i]];
              }
            } else {
              obj[columns[i].name] = record[i];
            }
          }
          if (raw === true || info2 === true) {
            const extRecord = Object.assign(
              { record: obj },
              raw === true ? { raw: this.state.rawBuffer.toString(encoding) } : {},
              info2 === true ? { info: this.__infoRecord() } : {}
            );
            const err = this.__push(
              objname === void 0 ? extRecord : [obj[objname], extRecord],
              push
            );
            if (err) {
              return err;
            }
          } else {
            const err = this.__push(
              objname === void 0 ? obj : [obj[objname], obj],
              push
            );
            if (err) {
              return err;
            }
          }
        } else {
          if (raw === true || info2 === true) {
            const extRecord = Object.assign(
              { record },
              raw === true ? { raw: this.state.rawBuffer.toString(encoding) } : {},
              info2 === true ? { info: this.__infoRecord() } : {}
            );
            const err = this.__push(
              objname === void 0 ? extRecord : [record[objname], extRecord],
              push
            );
            if (err) {
              return err;
            }
          } else {
            const err = this.__push(
              objname === void 0 ? record : [record[objname], record],
              push
            );
            if (err) {
              return err;
            }
          }
        }
      }
      this.__resetRecord();
    },
    __firstLineToColumns: function(record) {
      const { firstLineToHeaders } = this.state;
      try {
        const headers = firstLineToHeaders === void 0 ? record : firstLineToHeaders.call(null, record);
        if (!Array.isArray(headers)) {
          return this.__error(
            new CsvError(
              "CSV_INVALID_COLUMN_MAPPING",
              [
                "Invalid Column Mapping:",
                "expect an array from column function,",
                `got ${JSON.stringify(headers)}`
              ],
              this.options,
              this.__infoField(),
              {
                headers
              }
            )
          );
        }
        const normalizedHeaders = normalize_columns_array(headers);
        this.state.expectedRecordLength = normalizedHeaders.length;
        this.options.columns = normalizedHeaders;
        this.__resetRecord();
        return;
      } catch (err) {
        return err;
      }
    },
    __resetRecord: function() {
      if (this.options.raw === true) {
        this.state.rawBuffer.reset();
      }
      this.state.error = void 0;
      this.state.record = [];
      this.state.record_length = 0;
    },
    __onField: function() {
      const { cast, encoding, rtrim, max_record_size } = this.options;
      const { enabled, wasQuoting } = this.state;
      if (enabled === false) {
        return this.__resetField();
      }
      let field = this.state.field.toString(encoding);
      if (rtrim === true && wasQuoting === false) {
        field = field.trimRight();
      }
      if (cast === true) {
        const [err, f] = this.__cast(field);
        if (err !== void 0)
          return err;
        field = f;
      }
      this.state.record.push(field);
      if (max_record_size !== 0 && typeof field === "string") {
        this.state.record_length += field.length;
      }
      this.__resetField();
    },
    __resetField: function() {
      this.state.field.reset();
      this.state.wasQuoting = false;
    },
    __push: function(record, push) {
      const { on_record } = this.options;
      if (on_record !== void 0) {
        const info2 = this.__infoRecord();
        try {
          record = on_record.call(null, record, info2);
        } catch (err) {
          return err;
        }
        if (record === void 0 || record === null) {
          return;
        }
      }
      push(record);
    },
    // Return a tuple with the error and the casted value
    __cast: function(field) {
      const { columns, relax_column_count } = this.options;
      const isColumns = Array.isArray(columns);
      if (isColumns === true && relax_column_count && this.options.columns.length <= this.state.record.length) {
        return [void 0, void 0];
      }
      if (this.state.castField !== null) {
        try {
          const info2 = this.__infoField();
          return [void 0, this.state.castField.call(null, field, info2)];
        } catch (err) {
          return [err];
        }
      }
      if (this.__isFloat(field)) {
        return [void 0, parseFloat(field)];
      } else if (this.options.cast_date !== false) {
        const info2 = this.__infoField();
        return [void 0, this.options.cast_date.call(null, field, info2)];
      }
      return [void 0, field];
    },
    // Helper to test if a character is a space or a line delimiter
    __isCharTrimable: function(buf, pos) {
      const isTrim = (buf2, pos2) => {
        const { timchars } = this.state;
        loop1:
          for (let i = 0; i < timchars.length; i++) {
            const timchar = timchars[i];
            for (let j = 0; j < timchar.length; j++) {
              if (timchar[j] !== buf2[pos2 + j])
                continue loop1;
            }
            return timchar.length;
          }
        return 0;
      };
      return isTrim(buf, pos);
    },
    // Keep it in case we implement the `cast_int` option
    // __isInt(value){
    //   // return Number.isInteger(parseInt(value))
    //   // return !isNaN( parseInt( obj ) );
    //   return /^(\-|\+)?[1-9][0-9]*$/.test(value)
    // }
    __isFloat: function(value) {
      return value - parseFloat(value) + 1 >= 0;
    },
    __compareBytes: function(sourceBuf, targetBuf, targetPos, firstByte) {
      if (sourceBuf[0] !== firstByte)
        return 0;
      const sourceLength = sourceBuf.length;
      for (let i = 1; i < sourceLength; i++) {
        if (sourceBuf[i] !== targetBuf[targetPos + i])
          return 0;
      }
      return sourceLength;
    },
    __isDelimiter: function(buf, pos, chr) {
      const { delimiter, ignore_last_delimiters } = this.options;
      if (ignore_last_delimiters === true && this.state.record.length === this.options.columns.length - 1) {
        return 0;
      } else if (ignore_last_delimiters !== false && typeof ignore_last_delimiters === "number" && this.state.record.length === ignore_last_delimiters - 1) {
        return 0;
      }
      loop1:
        for (let i = 0; i < delimiter.length; i++) {
          const del = delimiter[i];
          if (del[0] === chr) {
            for (let j = 1; j < del.length; j++) {
              if (del[j] !== buf[pos + j])
                continue loop1;
            }
            return del.length;
          }
        }
      return 0;
    },
    __isRecordDelimiter: function(chr, buf, pos) {
      const { record_delimiter } = this.options;
      const recordDelimiterLength = record_delimiter.length;
      loop1:
        for (let i = 0; i < recordDelimiterLength; i++) {
          const rd = record_delimiter[i];
          const rdLength = rd.length;
          if (rd[0] !== chr) {
            continue;
          }
          for (let j = 1; j < rdLength; j++) {
            if (rd[j] !== buf[pos + j]) {
              continue loop1;
            }
          }
          return rd.length;
        }
      return 0;
    },
    __isEscape: function(buf, pos, chr) {
      const { escape } = this.options;
      if (escape === null)
        return false;
      const l = escape.length;
      if (escape[0] === chr) {
        for (let i = 0; i < l; i++) {
          if (escape[i] !== buf[pos + i]) {
            return false;
          }
        }
        return true;
      }
      return false;
    },
    __isQuote: function(buf, pos) {
      const { quote } = this.options;
      if (quote === null)
        return false;
      const l = quote.length;
      for (let i = 0; i < l; i++) {
        if (quote[i] !== buf[pos + i]) {
          return false;
        }
      }
      return true;
    },
    __autoDiscoverRecordDelimiter: function(buf, pos) {
      const { encoding } = this.options;
      const rds = [
        // Important, the windows line ending must be before mac os 9
        Buffer.from("\r\n", encoding),
        Buffer.from("\n", encoding),
        Buffer.from("\r", encoding)
      ];
      loop:
        for (let i = 0; i < rds.length; i++) {
          const l = rds[i].length;
          for (let j = 0; j < l; j++) {
            if (rds[i][j] !== buf[pos + j]) {
              continue loop;
            }
          }
          this.options.record_delimiter.push(rds[i]);
          this.state.recordDelimiterMaxLength = rds[i].length;
          return rds[i].length;
        }
      return 0;
    },
    __error: function(msg) {
      const { encoding, raw, skip_records_with_error } = this.options;
      const err = typeof msg === "string" ? new Error(msg) : msg;
      if (skip_records_with_error) {
        this.state.recordHasError = true;
        if (this.options.on_skip !== void 0) {
          this.options.on_skip(
            err,
            raw ? this.state.rawBuffer.toString(encoding) : void 0
          );
        }
        return void 0;
      } else {
        return err;
      }
    },
    __infoDataSet: function() {
      return {
        ...this.info,
        columns: this.options.columns
      };
    },
    __infoRecord: function() {
      const { columns, raw, encoding } = this.options;
      return {
        ...this.__infoDataSet(),
        error: this.state.error,
        header: columns === true,
        index: this.state.record.length,
        raw: raw ? this.state.rawBuffer.toString(encoding) : void 0
      };
    },
    __infoField: function() {
      const { columns } = this.options;
      const isColumns = Array.isArray(columns);
      return {
        ...this.__infoRecord(),
        column: isColumns === true ? columns.length > this.state.record.length ? columns[this.state.record.length].name : null : this.state.record.length,
        quoting: this.state.wasQuoting
      };
    }
  };
};
var parse = function(data, opts = {}) {
  if (typeof data === "string") {
    data = Buffer.from(data);
  }
  const records = opts && opts.objname ? {} : [];
  const parser = transform(opts);
  const push = (record) => {
    if (parser.options.objname === void 0)
      records.push(record);
    else {
      records[record[0]] = record[1];
    }
  };
  const close = () => {
  };
  const err1 = parser.parse(data, false, push, close);
  if (err1 !== void 0)
    throw err1;
  const err2 = parser.parse(void 0, true, push, close);
  if (err2 !== void 0)
    throw err2;
  return records;
};

// ../../node_modules/myst-directives/dist/table.js
var tableDirective = {
  name: "table",
  arg: {
    type: "myst",
    doc: "An optional table caption"
  },
  options: {
    ...commonDirectiveOptions("table"),
    class: {
      type: String,
      // class_option: list of strings?
      doc: `CSS classes to add to your table. Special classes include:

- \`full-width\`: changes the table environment to cover two columns in LaTeX`
    },
    align: {
      type: String
      // choice(['left', 'center', 'right'])
    }
  },
  body: {
    type: "myst",
    required: true
  },
  run(data) {
    const children = [];
    if (data.arg) {
      children.push({
        type: "caption",
        children: [{ type: "paragraph", children: data.arg }]
      });
    }
    children.push(...data.body);
    const container = {
      type: "container",
      kind: "table",
      children
    };
    addCommonDirectiveOptions(data, container);
    return [container];
  }
};
var listTableDirective = {
  name: "list-table",
  arg: {
    type: "myst",
    doc: "An optional table caption"
  },
  options: {
    ...commonDirectiveOptions("list table"),
    "header-rows": {
      type: Number
      // nonnegative int
    },
    // 'stub-columns': {
    //   type: Number,
    //   // nonnegative int
    // },
    // width: {
    //   type: String,
    //   // length_or_percentage_or_unitless,
    // },
    // widths: {
    //   type: String,
    //   // TODO use correct widths option validator
    // },
    class: {
      type: String,
      // class_option: list of strings?
      doc: `CSS classes to add to your table. Special classes include:

- \`full-width\`: changes the table environment to cover two columns in LaTeX`
    },
    align: {
      type: String
      // choice(['left', 'center', 'right'])
    }
  },
  body: {
    type: "myst",
    required: true
  },
  validate(data, vfile) {
    var _a;
    const validatedData = { ...data };
    const parsedBody = data.body;
    if (parsedBody.length !== 1 || parsedBody[0].type !== "list") {
      fileError(vfile, "list-table directive must have one list as body", {
        node: data.node,
        ruleId: RuleId.directiveBodyCorrect
      });
      validatedData.body = [];
    } else {
      (_a = parsedBody[0].children) === null || _a === void 0 ? void 0 : _a.forEach((listItem) => {
        var _a2, _b;
        if (!validatedData.body.length)
          return;
        if (listItem.type !== "listItem" || ((_a2 = listItem.children) === null || _a2 === void 0 ? void 0 : _a2.length) !== 1 || ((_b = listItem.children[0]) === null || _b === void 0 ? void 0 : _b.type) !== "list") {
          fileError(vfile, "list-table directive must have a list of lists", {
            node: data.node,
            ruleId: RuleId.directiveBodyCorrect
          });
          validatedData.body = [];
        }
      });
    }
    return validatedData;
  },
  run(data) {
    var _a, _b, _c;
    const children = [];
    if (data.arg) {
      children.push({
        type: "caption",
        children: [{ type: "paragraph", children: data.arg }]
      });
    }
    const topListChildren = ((_a = data.body[0]) === null || _a === void 0 ? void 0 : _a.children) || [];
    let headerCount = ((_b = data.options) === null || _b === void 0 ? void 0 : _b["header-rows"]) || 0;
    const table = {
      type: "table",
      align: (_c = data.options) === null || _c === void 0 ? void 0 : _c.align,
      children: topListChildren.map((topListItem) => {
        var _a2, _b2;
        const nestedListChildren = ((_b2 = (_a2 = topListItem.children) === null || _a2 === void 0 ? void 0 : _a2[0]) === null || _b2 === void 0 ? void 0 : _b2.children) || [];
        const row = {
          type: "tableRow",
          children: nestedListChildren.map((nestedListItem) => {
            const cell = {
              type: "tableCell",
              header: headerCount > 0 ? true : void 0,
              children: nestedListItem.children
            };
            return cell;
          })
        };
        headerCount -= 1;
        return row;
      })
    };
    children.push(table);
    const container = {
      type: "container",
      kind: "table",
      children
    };
    addCommonDirectiveOptions(data, container);
    return [container];
  }
};
function parseCSV(data, ctx, opts) {
  var _a, _b, _c;
  const delimiter = (_a = opts === null || opts === void 0 ? void 0 : opts.delim) !== null && _a !== void 0 ? _a : ",";
  const records = parse(data, {
    delimiter: delimiter === "tab" ? "	" : delimiter === "space" ? " " : delimiter,
    ltrim: !(opts === null || opts === void 0 ? void 0 : opts.keepspace),
    escape: (_b = opts === null || opts === void 0 ? void 0 : opts.escape) !== null && _b !== void 0 ? _b : '"',
    quote: (_c = opts === null || opts === void 0 ? void 0 : opts.quote) !== null && _c !== void 0 ? _c : '"'
  });
  return records.map((record, recordIndex) => {
    return record.map((cell) => {
      const mdast = ctx.parseMyst(cell, recordIndex);
      return select("*:root > paragraph:only-child", mdast);
    });
  });
}
var csvTableDirective = {
  name: "csv-table",
  doc: 'The "csv-table" directive is used to create a table from CSV (comma-separated values) data.',
  arg: {
    type: "myst",
    doc: "An optional table caption"
  },
  options: {
    ...commonDirectiveOptions("CSV table"),
    // file: {
    //   type: String,
    //   doc: 'The local filesystem path to a CSV data file.',
    //   alias: ['url'],
    //  Add this to the description for the directive:
    //  The data may be internal (an integral part of the document) or external (a separate file).
    // },
    header: {
      type: String,
      // nonnegative int
      doc: "Supplemental data for the table header, added independently of and before any header-rows from the main CSV data. Must use the same CSV format as the main CSV data."
    },
    "header-rows": {
      type: Number,
      // nonnegative int
      doc: "The number of rows of CSV data to use in the table header. Defaults to 0."
    },
    class: {
      type: String,
      // class_option: list of strings?
      doc: `CSS classes to add to your table. Special classes include:

- \`full-width\`: changes the table environment to cover two columns in LaTeX`
    },
    align: {
      type: String
      // choice(['left', 'center', 'right'])
    },
    delim: {
      type: String,
      doc: 'The character used to separate data fields. The special values "tab" and "space" are converted to the respective whitespace characters. Defaults to "," (comma)'
    },
    keepspace: {
      type: Boolean,
      doc: "Treat whitespace immediately following the delimiter as significant. The default is to ignore such whitespace."
    },
    quote: {
      type: String,
      doc: 'The character used to quote fields containing special characters, such as the delimiter, quotes, or new-line characters. Must be a single character, defaults to `"` (a double quote)\\\nFor example, `First cell, "These commas, for example, are escaped", Next cell`'
    },
    escape: {
      type: String,
      doc: 'A character used to escape the delimiter or quote characters from the CSV parser. Must be a single character, defaults to `"` (a double quote) default is a double quote\\\nFor example, `First cell, "These quotes"", for example, are escaped", Next cell`'
    }
  },
  body: {
    type: String,
    doc: "The CSV content",
    required: true
  },
  run(data, vfile, ctx) {
    var _a, _b, _c, _d, _e;
    const captions = [];
    if (data.arg) {
      captions.push({
        type: "caption",
        children: [{ type: "paragraph", children: data.arg }]
      });
    }
    const rows = [];
    if (((_a = data.options) === null || _a === void 0 ? void 0 : _a.header) !== void 0) {
      let headerCells = [];
      try {
        headerCells = parseCSV(data.options.header, ctx, data.options);
      } catch (error) {
        fileError(vfile, "csv-table directive header must be valid CSV-formatted MyST", {
          node: (_b = select('mystOption[name="header"]', data.node)) !== null && _b !== void 0 ? _b : data.node,
          ruleId: RuleId.directiveOptionsCorrect
        });
      }
      rows.push(...headerCells.map((parsedRow) => ({
        type: "tableRow",
        children: parsedRow.map((parsedCell) => {
          var _a2;
          return {
            type: "tableCell",
            header: true,
            children: (_a2 = parsedCell === null || parsedCell === void 0 ? void 0 : parsedCell.children) !== null && _a2 !== void 0 ? _a2 : []
          };
        })
      })));
    }
    let bodyCells = [];
    try {
      bodyCells = parseCSV(data.body, ctx, data.options);
    } catch (error) {
      fileError(vfile, "csv-table directive body must be valid CSV-formatted MyST", {
        node: (_c = select("mystDirectiveBody", data.node)) !== null && _c !== void 0 ? _c : data.node,
        ruleId: RuleId.directiveBodyCorrect
      });
    }
    let headerCount = ((_d = data.options) === null || _d === void 0 ? void 0 : _d["header-rows"]) || 0;
    rows.push(...bodyCells.map((parsedRow) => {
      const row = {
        type: "tableRow",
        children: parsedRow.map((parsedCell) => {
          var _a2;
          return {
            type: "tableCell",
            header: headerCount > 0 ? true : void 0,
            children: (_a2 = parsedCell === null || parsedCell === void 0 ? void 0 : parsedCell.children) !== null && _a2 !== void 0 ? _a2 : []
          };
        })
      };
      headerCount -= 1;
      return row;
    }));
    const table = {
      type: "table",
      align: (_e = data.options) === null || _e === void 0 ? void 0 : _e.align,
      children: rows
    };
    const container = {
      type: "container",
      kind: "table",
      children: [...captions, table]
    };
    addCommonDirectiveOptions(data, container);
    return [container];
  }
};

// ../../node_modules/myst-directives/dist/aside.js
var asideDirective = {
  name: "aside",
  alias: ["margin", "sidebar", "topic"],
  arg: {
    type: "myst",
    doc: "An optional title"
  },
  options: {
    ...commonDirectiveOptions("aside")
  },
  body: {
    type: "myst",
    required: true
  },
  run(data) {
    const children = [...data.body];
    if (data.arg) {
      children.unshift({
        type: "admonitionTitle",
        children: data.arg
      });
    }
    const aside = {
      type: "aside",
      kind: data.name == "aside" || data.name == "margin" ? void 0 : data.name,
      children
    };
    addCommonDirectiveOptions(data, aside);
    return [aside];
  }
};

// ../../node_modules/myst-directives/dist/glossary.js
var glossaryDirective = {
  name: "glossary",
  body: {
    type: "myst",
    required: true
  },
  options: {
    ...commonDirectiveOptions("glossary")
  },
  run(data) {
    const glossary = {
      type: "glossary",
      children: data.body
    };
    addCommonDirectiveOptions(data, glossary);
    return [glossary];
  }
};

// ../../node_modules/myst-directives/dist/math.js
var mathDirective = {
  name: "math",
  options: {
    ...commonDirectiveOptions("math"),
    typst: {
      type: String,
      doc: "Typst-specific math content. If not provided, LaTeX content will be converted to Typst."
    }
  },
  body: {
    type: String,
    required: true
  },
  run(data) {
    var _a;
    const math = addCommonDirectiveOptions(data, { type: "math", value: data.body });
    if (data.node.tight) {
      math.tight = data.node.tight;
    }
    if ((_a = data.options) === null || _a === void 0 ? void 0 : _a.typst) {
      math.typst = data.options.typst;
    }
    return [math];
  }
};

// ../../node_modules/myst-directives/dist/mdast.js
var mdastDirective = {
  name: "mdast",
  arg: {
    type: String,
    required: true
  },
  options: {
    ...commonDirectiveOptions("mdast")
  },
  run(data) {
    const mdast = {
      type: "mdast",
      id: data.arg
    };
    addCommonDirectiveOptions(data, mdast);
    return [mdast];
  }
};

// ../../node_modules/myst-directives/dist/mermaid.js
var mermaidDirective = {
  name: "mermaid",
  options: {
    ...commonDirectiveOptions("mermaid")
  },
  body: {
    type: String,
    required: true
  },
  run(data) {
    return [
      addCommonDirectiveOptions(data, {
        type: "mermaid",
        value: data.body
      })
    ];
  }
};

// ../../node_modules/myst-directives/dist/mystdemo.js
var mystdemoDirective = {
  name: "myst",
  doc: "Demonstrate some MyST code in an editable code block that displays its output/result interactively. Limited to built-in functionality, as parsing and rendering is done in the browser without access to plugin code or project configuration.",
  options: {
    ...commonDirectiveOptions("myst"),
    numbering: {
      type: String
    }
  },
  body: {
    type: String,
    required: true
  },
  run(data) {
    var _a;
    let numbering;
    if ((_a = data.options) === null || _a === void 0 ? void 0 : _a.numbering) {
      try {
        numbering = jsYaml.load(data.options.numbering);
      } catch (err) {
      }
    }
    const myst = {
      type: "myst",
      numbering,
      value: data.body
    };
    addCommonDirectiveOptions(data, myst);
    return [myst];
  }
};

// ../../node_modules/myst-directives/dist/blockquote.js
var import_classnames = __toESM(require_classnames(), 1);
var blockquoteDirective = {
  name: "blockquote",
  alias: ["epigraph", "pull-quote"],
  doc: "Block quotes are used to indicate that the enclosed content forms an extended quotation. They may be followed by an inscription or attribution formed of a paragraph beginning with `--`, `---`, or an em-dash.",
  options: {
    ...commonDirectiveOptions("blockquote"),
    // TODO: Add enumeration in future
    class: {
      type: String,
      doc: `CSS classes to add to your blockquote. Special classes include:

- \`pull-quote\`: used for a blockquote node which should attract attention
- \`epigraph\`: used for a blockquote node that are usually found at the beginning of a document`
    }
  },
  body: {
    type: "myst",
    doc: "The body of the quote, which may contain a special attribution paragraph that is turned into a caption"
  },
  run(data) {
    var _a;
    const children = [];
    if (data.body) {
      children.push(...data.body);
    }
    const container = {
      type: "container",
      kind: "quote",
      children: [
        {
          // @ts-expect-error: myst-spec needs updating to support blockquote
          type: "blockquote",
          children
        }
      ]
    };
    addCommonDirectiveOptions(data, container);
    const className = (_a = data.options) === null || _a === void 0 ? void 0 : _a.class;
    container.class = (0, import_classnames.default)({
      [className]: className,
      [data.name]: data.name !== "blockquote"
    });
    return [container];
  }
};

// ../../node_modules/myst-directives/dist/raw.js
var rawDirective = {
  name: "raw",
  doc: 'Allows you to include non-markdown text in your document. If the argument "latex" is provided, the text will be parsed as latex; otherwise, it will be included as raw text.',
  arg: {
    type: String,
    doc: 'Format of directive content - for now, only "latex" is valid'
  },
  body: {
    type: String,
    doc: "Raw content to be parsed"
  },
  run(data) {
    var _a, _b;
    const lang = (_a = data.arg) !== null && _a !== void 0 ? _a : "";
    const value = (_b = data.body) !== null && _b !== void 0 ? _b : "";
    const tex = ["tex", "latex"].includes(lang) ? `
${value}
` : void 0;
    const typst = ["typst", "typ"].includes(lang) ? `
${value}
` : void 0;
    return [
      {
        type: "raw",
        lang,
        tex,
        typst,
        value
      }
    ];
  }
};
var rawLatexDirective = {
  name: "raw:latex",
  alias: ["raw:tex"],
  doc: "Allows you to include tex in your document that will only be included in tex exports",
  body: {
    type: String,
    doc: "Raw tex content"
  },
  run(data) {
    const lang = "tex";
    const body = data.body;
    const tex = body ? `
${body}
` : "";
    return [
      {
        type: "raw",
        lang,
        tex
      }
    ];
  }
};
var rawTypstDirective = {
  name: "raw:typst",
  alias: ["raw:typ"],
  doc: "Allows you to include typst in your document that will only be included in typst exports",
  body: {
    type: String,
    doc: "Raw typst content"
  },
  run(data) {
    const lang = "typst";
    const body = data.body;
    const typst = body ? `
${body}
` : "";
    return [
      {
        type: "raw",
        lang,
        typst
      }
    ];
  }
};

// ../../node_modules/myst-directives/dist/div.js
var divDirective = {
  name: "div",
  options: {
    ...commonDirectiveOptions("div")
  },
  body: {
    type: "myst"
  },
  run(data) {
    const div = { type: "div" };
    if (data.body) {
      div.children = data.body;
    }
    addCommonDirectiveOptions(data, div);
    return [div];
  }
};

// ../../node_modules/myst-directives/dist/toc.js
var CONTEXTS = ["project", "page", "section"];
var tocDirective = {
  name: "toc",
  doc: "Inserts table of contents in the page. This may be for the project (each page has an entry), the current page (each heading has an entry), or the current section (only headings in the section have an entry).",
  alias: ["tableofcontents", "table-of-contents", "toctree", "contents"],
  arg: {
    type: "myst",
    doc: "Heading to be included with table of contents"
  },
  options: {
    context: {
      type: String,
      doc: "Table of Contents context; one of project, page, or section",
      alias: ["kind"]
    },
    depth: {
      type: Number,
      doc: "Number of levels to include in Table of Contents; by default, all levels will be included",
      alias: ["maxdepth"]
    },
    ...commonDirectiveOptions("toc")
  },
  run(data, vfile) {
    var _a, _b, _c;
    let context = ((_a = data.options) === null || _a === void 0 ? void 0 : _a.context) ? data.options.context : data.name === "contents" ? "section" : "project";
    if (!CONTEXTS.includes(context)) {
      fileError(vfile, `Unknown context for ${data.name} directive: ${context}`);
      context = "project";
    }
    let depth = (_b = data.options) === null || _b === void 0 ? void 0 : _b.depth;
    if (depth != null && depth < 1) {
      fileError(vfile, `Table of Contents 'depth' must be a number greater than 0`);
      depth = void 0;
    }
    const children = [];
    if (data.arg) {
      const parsedArg = data.arg;
      if (((_c = parsedArg[0]) === null || _c === void 0 ? void 0 : _c.type) === "heading") {
        children.push(...parsedArg);
      } else {
        children.push({
          type: "heading",
          depth: 2,
          enumerated: false,
          children: parsedArg
        });
      }
    }
    const toc = { type: "toc", kind: context, depth, children };
    addCommonDirectiveOptions(data, toc);
    return [toc];
  }
};

// ../../node_modules/myst-directives/dist/index.js
var defaultDirectives = [
  admonitionDirective,
  bibliographyDirective,
  csvTableDirective,
  codeDirective,
  codeCellDirective,
  dropdownDirective,
  embedDirective,
  blockquoteDirective,
  figureDirective,
  iframeDirective,
  imageDirective,
  includeDirective,
  indexDirective,
  genIndexDirective,
  tableDirective,
  listTableDirective,
  asideDirective,
  glossaryDirective,
  mathDirective,
  mdastDirective,
  mermaidDirective,
  mystdemoDirective,
  rawDirective,
  rawLatexDirective,
  rawTypstDirective,
  divDirective,
  tocDirective
];

export {
  commonDirectiveOptions,
  addCommonDirectiveOptions,
  defaultDirectives
};
//# sourceMappingURL=/build/_shared/chunk-2GZXDR27.js.map
