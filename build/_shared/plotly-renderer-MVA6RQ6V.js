import {
  Widget,
  init_index_es62 as init_index_es6
} from "/build/_shared/chunk-ZNGG4FXY.js";
import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/jupyterlab-plotly/lib/plotly-renderer.js
init_index_es6();
var __awaiter = function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
var CSS_CLASS = "jp-RenderedPlotly";
var CSS_ICON_CLASS = "jp-MaterialIcon jp-PlotlyIcon";
var MIME_TYPE = "application/vnd.plotly.v1+json";
var RenderedPlotly = class extends Widget {
  /**
   * Create a new widget for rendering Plotly.
   */
  constructor(options) {
    super();
    this.addClass(CSS_CLASS);
    this._mimeType = options.mimeType;
    this._img_el = document.createElement("img");
    this._img_el.className = "plot-img";
    this.node.appendChild(this._img_el);
    this._img_el.addEventListener("mouseenter", (event) => {
      this.createGraph(this._model);
    });
  }
  /**
   * Render Plotly into this widget's node.
   */
  renderModel(model) {
    if (this.hasGraphElement()) {
      return Promise.resolve();
    }
    this._model = model;
    const png_data = model.data["image/png"];
    if (png_data !== void 0 && png_data !== null) {
      this.updateImage(png_data);
      return Promise.resolve();
    } else {
      return this.createGraph(model);
    }
  }
  hasGraphElement() {
    return this.node.querySelector(".plot-container") !== null;
  }
  updateImage(png_data) {
    this.hideGraph();
    this._img_el.src = "data:image/png;base64," + png_data;
    this.showImage();
  }
  hideGraph() {
    let el = this.node.querySelector(".plot-container");
    if (el !== null && el !== void 0) {
      el.style.display = "none";
    }
  }
  showGraph() {
    let el = this.node.querySelector(".plot-container");
    if (el !== null && el !== void 0) {
      el.style.display = "block";
    }
  }
  hideImage() {
    let el = this.node.querySelector(".plot-img");
    if (el !== null && el !== void 0) {
      el.style.display = "none";
    }
  }
  showImage() {
    let el = this.node.querySelector(".plot-img");
    if (el !== null && el !== void 0) {
      el.style.display = "block";
    }
  }
  createGraph(model) {
    const { data, layout, frames, config } = model.data[this._mimeType];
    if (!layout.height) {
      layout.height = 360;
    }
    const loadPlotly = () => __awaiter(this, void 0, void 0, function* () {
      if (RenderedPlotly.Plotly === null) {
        RenderedPlotly.Plotly = yield import("/build/_shared/plotly-ELWLZ5EC.js");
        RenderedPlotly.Plotly = RenderedPlotly.Plotly.default;
        RenderedPlotly._resolveLoadingPlotly();
      }
      return RenderedPlotly.loadingPlotly;
    });
    return loadPlotly().then(() => {
      return RenderedPlotly.Plotly.react(this.node, data, layout, config);
    }).then((plot) => {
      this.showGraph();
      this.hideImage();
      this.update();
      if (frames) {
        RenderedPlotly.Plotly.addFrames(this.node, frames);
      }
      if (this.node.offsetWidth > 0 && this.node.offsetHeight > 0) {
        RenderedPlotly.Plotly.toImage(plot, {
          format: "png",
          width: this.node.offsetWidth,
          height: this.node.offsetHeight
        }).then((url) => {
          const imageData = url.split(",")[1];
          if (model.data["image/png"] !== imageData) {
            model.setData({
              data: Object.assign(Object.assign({}, model.data), { "image/png": imageData })
            });
          }
        });
      }
      this.node.on("plotly_webglcontextlost", () => {
        const png_data = model.data["image/png"];
        if (png_data !== void 0 && png_data !== null) {
          this.updateImage(png_data);
          return Promise.resolve();
        }
      });
    });
  }
  /**
   * A message handler invoked on an `'after-show'` message.
   */
  onAfterShow(msg) {
    this.update();
  }
  /**
   * A message handler invoked on a `'resize'` message.
   */
  onResize(msg) {
    this.update();
  }
  /**
   * A message handler invoked on an `'update-request'` message.
   */
  onUpdateRequest(msg) {
    if (RenderedPlotly.Plotly && this.isVisible && this.hasGraphElement()) {
      RenderedPlotly.Plotly.redraw(this.node).then(() => {
        RenderedPlotly.Plotly.Plots.resize(this.node);
      });
    }
  }
};
RenderedPlotly.Plotly = null;
RenderedPlotly.loadingPlotly = new Promise((resolve) => {
  RenderedPlotly._resolveLoadingPlotly = resolve;
});
var rendererFactory = {
  safe: true,
  mimeTypes: [MIME_TYPE],
  createRenderer: (options) => new RenderedPlotly(options)
};
var extensions = [
  {
    id: "@jupyterlab/plotly-extension:factory",
    rendererFactory,
    rank: 0,
    dataType: "json",
    fileTypes: [
      {
        name: "plotly",
        mimeTypes: [MIME_TYPE],
        extensions: [".plotly", ".plotly.json"],
        iconClass: CSS_ICON_CLASS
      }
    ],
    documentWidgetFactoryOptions: {
      name: "Plotly",
      primaryFileType: "plotly",
      fileTypes: ["plotly", "json"],
      defaultFor: ["plotly"]
    }
  }
];
var plotly_renderer_default = extensions;
export {
  MIME_TYPE,
  RenderedPlotly,
  plotly_renderer_default as default,
  rendererFactory
};
//# sourceMappingURL=/build/_shared/plotly-renderer-MVA6RQ6V.js.map
