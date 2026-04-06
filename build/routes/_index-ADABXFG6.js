import {
  KatexCSS,
  Page
} from "/build/_shared/chunk-XF6KJEBA.js";
import {
  getMetaTagsForArticle,
  require_loaders
} from "/build/_shared/chunk-QYHIDYQN.js";
import "/build/_shared/chunk-IW6XPN43.js";
import "/build/_shared/chunk-ZNGG4FXY.js";
import "/build/_shared/chunk-5XJWQ5WJ.js";
import "/build/_shared/chunk-SP2MKLPW.js";
import "/build/_shared/chunk-T7QTAV5N.js";
import "/build/_shared/chunk-2WW6JOYG.js";
import "/build/_shared/chunk-IMMBAB6Q.js";
import "/build/_shared/chunk-7SI5XUIQ.js";
import "/build/_shared/chunk-DOEQHMLC.js";
import "/build/_shared/chunk-ZNG4DSCS.js";
import "/build/_shared/chunk-MWXHQG25.js";
import "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import "/build/_shared/chunk-2RVQXRZB.js";
import "/build/_shared/chunk-3RNZ6DIW.js";
import {
  __toESM
} from "/build/_shared/chunk-CGOEG7L2.js";

// app/routes/_index.tsx
var import_loaders = __toESM(require_loaders());
var meta = ({ data, location }) => {
  var _a, _b, _c, _d, _e, _f, _g;
  if (!data)
    return [];
  const config = data.config;
  const project = data.project;
  return getMetaTagsForArticle({
    origin: "",
    url: location.pathname,
    title: (_a = config == null ? void 0 : config.title) != null ? _a : project.title,
    description: (_c = (_b = config.description) != null ? _b : project.description) != null ? _c : void 0,
    image: (_d = project.thumbnailOptimized || project.thumbnail) != null ? _d : void 0,
    keywords: (_f = (_e = config.keywords) != null ? _e : project.keywords) != null ? _f : [],
    twitter: (_g = config == null ? void 0 : config.options) == null ? void 0 : _g.twitter
  });
};
var links = () => [KatexCSS];
var index_default = Page;
export {
  index_default as default,
  links,
  meta
};
//# sourceMappingURL=/build/routes/_index-ADABXFG6.js.map
