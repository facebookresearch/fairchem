import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-ext-card/dist/index.js
var HEADER_REGEX = /((?<before>[\s\S]*?)\s+){0,1}\^\^\^(\s+(?<after>[\s\S]*)){0,1}/;
var cardDirective = {
  name: "card",
  alias: ["grid-item-card"],
  arg: {
    type: "myst",
    doc: "The title of the card, usually shown as bolded text at the top of the card."
  },
  options: {
    url: {
      type: String,
      alias: ["link"],
      doc: "Turns the card into a link, can be internal or external."
    },
    header: {
      type: "myst",
      doc: "Adds a header to the card."
    },
    footer: {
      type: "myst",
      doc: "Adds a footer to the card."
    }
    // // https://sphinx-design.readthedocs.io/en/furo-theme/cards.html#card-options
    // width
    // margin
    // padding
    // 'text-align'
    // 'img-top'
    // 'img-background'
    // 'img-bottom'
    // link
    // // This should be removed, it is picked up just as any other link that can also be a reference
    // 'link-type'
    // 'link-alt'
    // // This should just be a class that is recognized (similar to dropdown)
    // shadow
    // 'class-card'
    // // I feel like all of these should *not* be included.
    // // Instead us a css selector on `class`: for example, `.card.my-class > header { customCss: prop; }`
    // 'class-header'
    // 'class-body'
    // 'class-title'
    // 'class-footer'
    // 'class-img-top'
    // 'class-img-bottom'
    // // https://sphinx-design.readthedocs.io/en/furo-theme/grids.html#grid-item-card-options
    // columns
    // 'class-item' // This seems the same as `class-card`?
  },
  body: {
    type: "myst",
    required: true,
    doc: "Main body content of the card."
  },
  run(data) {
    const { url, header, footer } = data.options || {};
    let headerChildren;
    let bodyChildren;
    let footerChildren;
    if (header || footer) {
      headerChildren = header ? [{ type: "paragraph", children: header }] : [];
      bodyChildren = data.body;
      footerChildren = footer ? [{ type: "paragraph", children: footer }] : [];
    } else {
      const [beforeHeaderDelim, afterHeaderDelim] = splitChildrenOnDelim(data.body);
      headerChildren = afterHeaderDelim.length ? beforeHeaderDelim : [];
      const nonHeaderChildren = afterHeaderDelim.length ? afterHeaderDelim : beforeHeaderDelim;
      [bodyChildren, footerChildren] = splitChildrenOnBlockBreak(nonHeaderChildren);
    }
    const children = [];
    if (headerChildren.length) {
      children.push({
        type: "header",
        children: headerChildren
      });
    }
    if (data.arg) {
      children.push({
        type: "cardTitle",
        children: data.arg
      });
    }
    children.push(...bodyChildren);
    if (footerChildren.length) {
      children.push({
        type: "footer",
        children: footerChildren
      });
    }
    return [
      {
        type: "card",
        url,
        children
      }
    ];
  }
};
function splitParagraphNode(node) {
  var _a;
  const preChildren = [];
  const postChildren = [];
  let post = false;
  (_a = node.children) === null || _a === void 0 ? void 0 : _a.forEach((child) => {
    var _a2;
    if (post) {
      postChildren.push(child);
      return;
    }
    if (child.type !== "text" || !child.value) {
      preChildren.push(child);
      return;
    }
    const value = child.value;
    const match = HEADER_REGEX.exec(value);
    if (!match) {
      preChildren.push(child);
      return;
    }
    post = true;
    const { before, after } = (_a2 = match.groups) !== null && _a2 !== void 0 ? _a2 : {};
    if (before) {
      preChildren.push({ type: "text", value: before });
    }
    if (after) {
      postChildren.push({ type: "text", value: after });
    }
  });
  const output = {
    before: preChildren.length ? { ...node, children: preChildren } : null,
    after: postChildren.length ? { ...node, children: postChildren } : null,
    post
  };
  return output;
}
function splitChildrenOnDelim(children) {
  const preChildren = [];
  const postChildren = [];
  let post = false;
  children.forEach((child) => {
    if (post) {
      postChildren.push(child);
    } else if (child.type !== "paragraph") {
      preChildren.push(child);
    } else {
      const split = splitParagraphNode(child);
      const { before, after } = split;
      post = split.post;
      if (before) {
        preChildren.push(before);
      }
      if (after) {
        postChildren.push(after);
      }
    }
  });
  return [preChildren, postChildren];
}
function splitChildrenOnBlockBreak(children) {
  const preChildren = [];
  const postChildren = [];
  let post = false;
  children.forEach((child) => {
    if (post) {
      postChildren.push(child);
    } else if (child.type !== "blockBreak") {
      preChildren.push(child);
    } else {
      post = true;
    }
  });
  return [preChildren, postChildren];
}
export {
  cardDirective,
  splitParagraphNode
};
//# sourceMappingURL=/build/_shared/dist-CG3I3ZBM.js.map
